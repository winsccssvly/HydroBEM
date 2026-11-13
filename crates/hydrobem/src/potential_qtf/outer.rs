//! The free surface beyond the outer circle (theory §12.9): the forcing of every pair on the θ
//! grid of each radius of a radial Gauss rule to a large radius with a taper, integrated against ψ_j
//! (the indirect method) and against the outgoing radial functions at Ω (the moments of the direct
//! method, which its free-surface weights take through G's far-field fit).

use num_complex::Complex64;
use rayon::prelude::*;

use super::bessel::bessel_jy;
use super::far_field::{LOCAL_ORDERS, Local, Radial, outgoing_radial};
use super::forcing::{Values, Wave, forcing};
use super::{CZ, First, I, Radiation, Setup};
use crate::constants::PI;
use crate::env_hooks::validation_constant as vc;
use crate::error::Result;
use crate::progress::{SecondOrderStep, SolveContext};

/// The radial points (r, weight) beyond the outer circle R: Gauss panels of a quarter of
/// 2π/(3 k_s), k_s the largest wavenumber of the first order and the pair frequencies, to R + L, L
/// four times the longest first-order wavelength (at least 10 R), a cos² taper over its last 30 %,
/// the weight r dr. The integrand oscillates and decays slowly (theory §12.5), so the integral
/// is the limit of the radius and the taper ends it smoothly; the length, the taper fraction and
/// the panel width are fixed parameters of the method (theory §12.9, the outer integral).
#[allow(non_snake_case)] // the notation of the theory documents
fn radial_points(s: &Setup) -> Vec<(f64, f64)> {
    let R = s.z.outer_radius;
    let kmin = s.z.kf.iter().copied().fold(f64::INFINITY, f64::min);
    let L = (vc("OUTER_WAVELENGTHS") * 2.0 * PI / kmin).max(vc("OUTER_RADII") * R);
    let width = vc("OUTER_PANEL") * 2.0 * PI / (3.0 * s.z.ks_max);
    let taper = vc("OUTER_TAPER");
    let npanel = ((L / width).ceil() as usize).max(1);
    let (gx, gw) = crate::quadrature::gauss_legendre(6);
    let mut rad: Vec<(f64, f64)> = Vec::with_capacity(npanel * 6);
    for p in 0..npanel {
        let (a, bb) = (R + L * p as f64 / npanel as f64, R + L * (p + 1) as f64 / npanel as f64);
        for (x, w) in gx.iter().zip(&gw) {
            let rr = 0.5 * (a + bb) + 0.5 * (bb - a) * x;
            let t = ((rr - (R + (1.0 - taper) * L)) / (taper * L)).clamp(0.0, 1.0);
            let f = if t > 0.0 { (0.5 * PI * t).cos().powi(2) } else { 1.0 };
            rad.push((rr, 0.5 * (bb - a) * w * f * rr));
        }
    }
    rad
}

/// The θ grid beyond the outer circle: NT = 4M + 8 points, e^{i n θ_t} for |n| <= 2M + 1 (the
/// Hankel modes and the band-limited incident factors, [t][n + 2M + 1]) and (cos θ_t, sin θ_t).
struct Grid {
    nt: usize,
    nn: usize, // 2M + 1
    ne: usize, // 2 nn + 1
    et: Vec<Complex64>,
    cs: Vec<(f64, f64)>,
}

impl Grid {
    fn new(mm: usize) -> Grid {
        let nt = 4 * mm + 8;
        let nn = 2 * mm + 1;
        let ne = 2 * nn + 1;
        let mut et = vec![CZ; nt * ne];
        for t in 0..nt {
            let th = 2.0 * PI * t as f64 / nt as f64;
            for n in 0..ne {
                et[t * ne + n] = Complex64::from_polar(1.0, (n as f64 - nn as f64) * th);
            }
        }
        let cs = (0..nt).map(|t| (2.0 * PI * t as f64 / nt as f64).sin_cos()).map(|(s, c)| (c, s)).collect();
        Grid { nt, nn, ne, et, cs }
    }
    fn row(&self, t: usize) -> &[Complex64] {
        &self.et[t * self.ne..(t + 1) * self.ne]
    }
}

/// The fields at one radius beyond the outer circle: the disturbances (φ, ∇φ, γ) on the θ grid of
/// every (f, h), the band-limited incident factors e^{∓i k r cos(θ − β)} of every plane wave
/// [f][h][the plane wave][∓], ψ_j on the θ grid of every used pair frequency (the indirect method)
/// and the outgoing radial functions at each used Ω.
struct AtRadius {
    s: Vec<Vec<Values>>,
    incs: Vec<Vec<Complex64>>,
    psi: Vec<Option<Vec<Complex64>>>,
    rad_o: Vec<Option<Radial>>,
}

#[allow(non_snake_case)] // the notation of the theory documents
#[allow(clippy::needless_range_loop)] // index notation over the θ grid
fn at_radius(s: &Setup, first: &First, rad: Option<&Radiation>, gr: &Grid, rr: f64) -> AtRadius {
    let (g, R, M, n6, F, H) = (s.g, s.z.outer_radius, s.z.order, s.n6, s.nf, s.nh);
    let (NT, NN, NE) = (gr.nt, gr.nn, gr.ne);
    // the synthesized disturbances (φ, ∇φ, γ) on the θ grid of every (f, h)
    let syn = |far: &super::far_field::Far, radf: &Radial| {
        let md = far.modes(radf, g);
        let mut out = vec![(CZ, [CZ; 3], CZ); NT];
        for t in 0..NT {
            let row = gr.row(t);
            let (mut p, mut pr, mut pt, mut gm) = (CZ, CZ, CZ, CZ);
            for (i, m) in (-(M as i64)..=(M as i64)).enumerate() {
                let e = row[(m + NN as i64) as usize];
                p += md[i][0] * e;
                pr += md[i][1] * e;
                pt += m as f64 * md[i][0] * e;
                gm += md[i][2] * e;
            }
            let pt = I * pt / rr;
            let (c, sn) = gr.cs[t];
            out[t] = (p, [c * pr - sn * pt, sn * pr + c * pt, far.kk * p], gm);
        }
        out
    };
    let synp = |far: &super::far_field::Far, radf: &Radial| -> Vec<Complex64> {
        let md = far.modes(radf, g);
        (0..NT)
            .map(|t| {
                let row = gr.row(t);
                let mut p = CZ;
                for (i, m) in (-(M as i64)..=(M as i64)).enumerate() {
                    p += row[(m + NN as i64) as usize] * md[i][0];
                }
                p
            })
            .collect()
    };
    // the radial functions per frequency (the local basis depends on it, not on the heading)
    let radf: Vec<Radial> = (0..F).map(|f| first.far[f * H].radial(rr, R, M)).collect();
    let sv: Vec<Vec<Values>> = (0..F * H).map(|fh| syn(&first.far[fh], &radf[fh / H])).collect();
    // the band-limited incident factors e^{∓i k r cos(θ − β)} (|n| <= 2M + 1, or M given)
    let inc = |j: &[f64], beta: f64, sgn: f64| -> Vec<Complex64> {
        let mut c = vec![CZ; NE];
        let mut ip = Complex64::new(1.0, 0.0); // (sgn i)^n, n >= 0
        let si = Complex64::new(0.0, sgn);
        for n in 0..=if s.z.given_order { M } else { NN } {
            let jn = j[n];
            let jneg = if n % 2 == 1 { -jn } else { jn };
            // (sgn i)^(-n) = conj((sgn i)^n) for |sgn i| = 1
            c[NN + n] = ip * jn * Complex64::from_polar(1.0, -(n as f64) * beta);
            c[NN - n] = ip.conj() * jneg * Complex64::from_polar(1.0, n as f64 * beta);
            ip *= si;
        }
        (0..NT).map(|t| gr.row(t).iter().zip(&c).map(|(x, y)| x * y).sum()).collect()
    };
    let mut incs: Vec<Vec<Complex64>> = Vec::new();
    for f in 0..F {
        let (j, _) = bessel_jy(NN, s.z.kf[f] * rr);
        for hh in 0..H {
            for beta in s.wave(f, hh).headings() {
                incs.push(inc(&j, beta, -1.0));
                incs.push(inc(&j, beta, 1.0));
            }
        }
    }
    let NO = s.z.oms.len();
    let mut psi: Vec<Option<Vec<Complex64>>> = vec![None; NO * n6];
    let mut rad_o: Vec<Option<Radial>> = vec![None; NO];
    for o in 0..NO {
        if !s.used[o] {
            continue;
        }
        let ro = match rad {
            Some(rp) => {
                let ro = rp.far[o * n6].radial(rr, R, M);
                for jm in 0..n6 {
                    psi[o * n6 + jm] = Some(synp(&rp.far[o * n6 + jm], &ro));
                }
                ro
            }
            // the radial functions of an outgoing field at Ω (as Far::radial)
            None => outgoing_radial(s.z.ko[o], Local::of(s.z.oms[o] * s.z.oms[o] / g, s.h, R), rr, R, M),
        };
        rad_o[o] = Some(ro);
    }
    AtRadius { s: sv, incs, psi, rad_o }
}

/// The contribution of the radius (rr, wr) to the outer integral of work item wi: acc[j] the
/// integral against ψ_j (the indirect method), acc[n6 + m + M] and acc[n6 + 2M + 1 + m + M] the
/// moments against H_m(k_Ω r) e^{imθ} and L_m(r) e^{imθ} (the direct method).
#[allow(non_snake_case)] // the notation of the theory documents
#[allow(clippy::needless_range_loop)] // index notation over the θ grid and the modes
fn add_radius(s: &Setup, gr: &Grid, at: &AtRadius, wr: f64, wi: usize, acc: &mut [Complex64]) {
    let (g, M, n6, H) = (s.g, s.z.order, s.n6, s.nh);
    let (NT, NN, NE) = (gr.nt, gr.nn, gr.ne);
    let c0 = s.z.c0;
    let np = if s.z.wall.is_some() { 2 } else { 1 }; // the plane waves of an incident wave
    let (a, bb, ha, hb, sum) = s.work[wi];
    let (wa, wb) = (s.wave(a, ha), s.wave(bb, hb));
    let o = s.oidx(if sum { wa.w + wb.w } else { wa.w - wb.w });
    let Sa = &at.s[a * H + ha];
    let Sb = &at.s[bb * H + hb];
    // the incident multipliers of each plane wave with the phase at the centre
    let ph = |w: &Wave, beta: f64| Complex64::from_polar(1.0, -w.k * (c0[0] * beta.cos() + c0[1] * beta.sin()));
    let ma: Vec<_> = wa.headings().into_iter().map(|beta| wa.multipliers(beta, g, ph(&wa, beta))).collect();
    let mb: Vec<_> = wb.headings().into_iter().map(|beta| wb.multipliers(beta, g, ph(&wb, beta))).collect();
    // e^{-i k_a r cos} of each plane wave; its conjugate for b at a difference
    let ea = |p: usize| &at.incs[((a * H + ha) * np + p) * 2];
    let eb = |p: usize| &at.incs[((bb * H + hb) * np + p) * 2 + if sum { 0 } else { 1 }];
    // the forcing on the θ grid
    let ft: Vec<Complex64> = (0..NT)
        .map(|t| {
            let (pa, gva, gsa) = Sa[t];
            let (pb, gvb, gsb) = Sb[t];
            let mut f = forcing(wa.w, wb.w, g, sum, pa, &gva, gsa, pb, &gvb, gsb);
            for (p, (ia, ga_, gia)) in ma.iter().enumerate() {
                f += forcing(wa.w, wb.w, g, sum, *ia, ga_, *gia, pb, &gvb, gsb) * ea(p)[t];
            }
            for (p, (ib, gb_, gib)) in mb.iter().enumerate() {
                f += forcing(wa.w, wb.w, g, sum, pa, &gva, gsa, *ib, gb_, *gib) * eb(p)[t];
            }
            f
        })
        .collect();
    if s.ind {
        for j in 0..n6 {
            let Some(psi) = &at.psi[o * n6 + j] else { continue };
            let mut sm = CZ;
            for t in 0..NT {
                sm += psi[t] * ft[t];
            }
            acc[j] += wr * sm * (2.0 * PI / NT as f64);
        }
    }
    if s.dir {
        // the moments against H_m(k_Ω r) e^{imθ} and L_m(r) e^{imθ} (index n6 + m + M and
        // n6 + 2M + 1 + m + M)
        let (hv, _, lv) = at.rad_o[o].as_ref().expect("the radial functions of Ω");
        for (i, mm) in (-(M as i64)..=(M as i64)).enumerate() {
            let mut dsum = CZ;
            for t in 0..NT {
                dsum += gr.et[t * NE + (mm + NN as i64) as usize] * ft[t];
            }
            let dsum = dsum * (2.0 * PI / NT as f64) * wr;
            acc[n6 + i] += hv[i] * dsum;
            let ma = mm.unsigned_abs() as usize;
            if ma <= LOCAL_ORDERS {
                acc[n6 + 2 * M + 1 + i] += lv[ma][0] * dsum;
            }
        }
    }
}

/// The sums of the outer integral of a work item: its forces and, for the direct method, the
/// moments of the wave and of the local term per order |m| <= M.
pub(super) fn accumulated(s: &Setup) -> usize {
    s.n6 + if s.dir { 2 * (2 * s.z.order + 1) } else { 0 }
}

/// The radii of a batch of the outer integral and the bytes of the fields of a batch held at once:
/// about 64 MB, at least one radius and at most 64 (counted by the memory estimate).
pub(super) fn batch(s: &Setup) -> (usize, usize) {
    let nt = 4 * s.z.order + 8; // Grid::nt
    let per = 16 * nt * (s.nf * s.nh * (5 + 4) + if s.ind { s.z.oms.len() * s.n6 } else { 0 }) + 1;
    let batch = (64usize << 20).div_ceil(per).clamp(1, 64);
    (batch, batch * per)
}

/// The outer integral of every work item (add_radius), halved with a wall (the whole plane): the
/// radii in batches, the fields of each radius in parallel, then every work item's sum over the
/// radii in their order (one running sum each: the same numbers with any thread count). The
/// batches are the items of its second-order step, each on all the threads (shared by the
/// workers); a cancel request is checked before every batch and every radius.
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn integral(
    s: &Setup,
    first: &First,
    rad: Option<&Radiation>,
    ctx: &SolveContext,
) -> Result<Vec<Vec<Complex64>>> {
    let gr = Grid::new(s.z.order);
    let mut acc = vec![vec![CZ; accumulated(s)]; s.work.len()];
    let pts = if s.work.is_empty() { Vec::new() } else { radial_points(s) };
    let (batch, _) = batch(s);
    let step = ctx.step(SecondOrderStep::PotentialOuter, pts.len().div_ceil(batch));
    let r = (|| -> Result<()> {
        for (k, chunk) in pts.chunks(batch).enumerate() {
            let it = step.item(k, None);
            let r = (|| -> Result<()> {
                it.check_cancel()?;
                let done = std::sync::atomic::AtomicUsize::new(0);
                let at: Vec<AtRadius> = chunk
                    .par_iter()
                    .map(|&(rr, _)| {
                        it.check_cancel()?;
                        let x = at_radius(s, first, rad, &gr, rr);
                        it.report(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1, chunk.len() + 1);
                        Ok(x)
                    })
                    .collect::<Result<_>>()?;
                acc.par_iter_mut().enumerate().try_for_each(|(wi, a)| -> Result<()> {
                    it.check_cancel()?;
                    for (x, &(_, wr)) in at.iter().zip(chunk) {
                        add_radius(s, &gr, x, wr, wi, a);
                    }
                    Ok(())
                })
            })();
            it.end(&r);
            r?;
        }
        Ok(())
    })();
    step.end(r.is_ok());
    r?;
    if s.work.is_empty() {
        return Ok(acc);
    }
    for v in acc.iter_mut() {
        for x in v.iter_mut() {
            *x *= s.hf;
        }
    }
    Ok(acc)
}
