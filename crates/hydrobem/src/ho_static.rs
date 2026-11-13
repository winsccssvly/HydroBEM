//! The static (frequency-independent) integrals of the higher-order method: the Rankine source and
//! dipole of the image sum and the log-singularity terms of one target point against every
//! instance (theory §7).

use crate::bspline::basis;
use crate::geometry::{Vec3, dot, norm, v3};
use crate::ho_geometry::point_modes;
use crate::ho_internal::{
    FAR_ORDER, HoGeom, MAX_MODES, MID_ORDER, PatchDisc, Pt, StaticRow, WAVE_FAR, block_of, inst_eval,
};
use crate::quadrature::gauss_nodes;
use crate::tables::log_gauss_table::{LOG_GAUSS_N, LOG_GAUSS_NODES, LOG_GAUSS_WEIGHTS};

// The Rankine images of the target point x: x[q] the image points, si[q] their signs at
// omega = inf. At omega = 0 and in the ordinary problem every sign is +1, so none is stored.
// The same terms in the same order as rankine_images of ho_sheets.rs, which images the source
// point y instead (|x_q - y| = |x - y_q|: the image x + 2h of the target is the image y - 2h of
// the source, so the two tables list the translations in opposite order).
#[derive(Clone, Copy, Debug)]
struct Images {
    x: [Vec3; 6],
    si: [f64; 6],
    n: usize,
    xb: Vec3, // image of the target above the free surface (singularity of psi)
}

impl Default for Images {
    fn default() -> Self {
        Images { x: [Vec3::ZERO; 6], si: [0.0; 6], n: 2, xb: Vec3::ZERO }
    }
}

#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn images_of(x: &Vec3, h: f64) -> Images {
    let mut im = Images { xb: v3(x[0], x[1], -x[2]), ..Images::default() };
    if !h.is_finite() {
        im.n = 2;
        im.x[0] = *x;
        im.x[1] = im.xb;
        im.si[0] = 1.0;
        im.si[1] = -1.0;
        return im;
    }
    im.n = 6;
    let p = [
        *x,
        v3(x[0], x[1], -x[2] - 2.0 * h),
        v3(x[0], x[1], -x[2]),
        v3(x[0], x[1], x[2] + 2.0 * h),
        v3(x[0], x[1], x[2] - 2.0 * h),
        v3(x[0], x[1], -x[2] - 4.0 * h),
    ];
    for q in 0..6 {
        im.x[q] = p[q];
        im.si[q] = if q < 2 { 1.0 } else { -1.0 };
    }
    im
}

struct Sampler<'h, 'g, 'r> {
    h: &'h HoGeom<'g>,
    im: &'h Images,
    row: &'r mut StaticRow,
    want_inf: bool,
    logpart: bool,
}

impl Sampler<'_, '_, '_> {
    // one sample at parameters (u, v) of instance ii, panel (iu, iv), weight w (parametric)
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn sample(&mut self, ii: i32, iu: i32, iv: i32, u: f64, v: f64, w: f64) {
        let h = self.h;
        let im = self.im;
        let in_ = &h.inst[ii as usize];
        let d = &h.disc[in_.patch as usize];
        let (mut x, mut n, mut xb, mut nb) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let mut J = 0.0;
        inst_eval(h, in_, u, v, &mut x, &mut n, &mut J, Some(&mut xb), Some(&mut nb));
        let wJ = w * J;
        let (mut s, mut dd, mut si, mut ddi) = (0.0, 0.0, 0.0, 0.0);
        for q in 0..im.n {
            let r = im.x[q] - x;
            let r2 = dot(r, r);
            let ir = 1.0 / r2.sqrt();
            let ir3 = ir / r2;
            let nr = dot(n, r) * ir3;
            s += ir;
            dd += nr;
            si += im.si[q] * ir;
            ddi += im.si[q] * nr;
        }
        let (mut psi, mut dpsi) = (0.0, 0.0);
        if self.logpart {
            let dl = x - im.xb;
            let R = norm(dl);
            let den = R - dl[2];
            psi = den.ln();
            dpsi = (dot(dl, n) / R - n[2]) / den;
        }
        let (mut Nu, mut Nv) = ([0.0f64; 8], [0.0f64; 8]);
        basis(&d.Tu, d.ku, iu + d.ku - 1, u, &mut Nu, None);
        basis(&d.Tv, d.kv, iv + d.kv - 1, v, &mut Nv, None);
        let mu = d.mu();
        let c0 = h.col0[ii as usize];
        let row = &mut *self.row;
        for b in 0..d.kv {
            for a in 0..d.ku {
                let c = (c0 + (iv + b) * mu + iu + a) as usize;
                let wb = wJ * Nu[a as usize] * Nv[b as usize];
                row.rs[c] += s * wb;
                row.rd[c] += dd * wb;
                if self.want_inf {
                    row.rsi[c] += si * wb;
                    row.rdi[c] += ddi * wb;
                }
                if self.logpart {
                    row.ls[c] += psi * wb;
                    row.ld[c] += dpsi * wb;
                }
            }
        }
        let mut nm = [0.0f64; MAX_MODES];
        point_modes(h, ii, &xb, &nb, &mut nm);
        let NM = h.NM as usize;
        let base = ii as usize * NM;
        for k in 0..NM {
            if nm[k] == 0.0 {
                continue;
            }
            row.ms[base + k] += s * wJ * nm[k];
            if self.want_inf {
                row.msi[base + k] += si * wJ * nm[k];
            }
            if self.logpart {
                row.ml[base + k] += psi * wJ * nm[k];
            }
        }
    }
}

// The same as Sampler for stored points (area weights, basis values, normals).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn accumulate_points(h: &HoGeom, im: &Images, ii: i32, pts: &[Pt], want_inf: bool, logpart: bool, row: &mut StaticRow) {
    let d = &h.disc[h.inst[ii as usize].patch as usize];
    let mu = d.mu();
    let c0 = h.col0[ii as usize];
    for q in pts {
        let (mut sv, mut dv, mut svi, mut dvi) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..im.n {
            let r = im.x[k] - q.x;
            let r2 = dot(r, r);
            let ir = 1.0 / r2.sqrt();
            let nr = dot(q.n, r) * ir / r2;
            sv += ir;
            dv += nr;
            svi += im.si[k] * ir;
            dvi += im.si[k] * nr;
        }
        let (mut psi, mut dpsi) = (0.0, 0.0);
        if logpart {
            let dl = q.x - im.xb;
            let R = norm(dl);
            let den = R - dl[2];
            // a point on the image of the target (coincident points in z = 0, when the outer
            // and inner rules share a node): the wave kernel there is its regular limit
            // without psi, so psi is left out here too
            if den > 0.0 {
                psi = den.ln();
                dpsi = (dot(dl, q.n) / R - q.n[2]) / den;
            }
        }
        for b in 0..d.kv {
            for a in 0..d.ku {
                let c = (c0 + (q.iv + b) * mu + q.iu + a) as usize;
                let wb = q.w * q.bu[a as usize] * q.bv[b as usize];
                row.rs[c] += sv * wb;
                row.rd[c] += dv * wb;
                if want_inf {
                    row.rsi[c] += svi * wb;
                    row.rdi[c] += dvi * wb;
                }
                if logpart {
                    row.ls[c] += psi * wb;
                    row.ld[c] += dpsi * wb;
                }
            }
        }
        let NM = h.NM as usize;
        let base = ii as usize * NM;
        for k in 0..NM {
            if q.nm[k] == 0.0 {
                continue;
            }
            row.ms[base + k] += sv * q.w * q.nm[k];
            if want_inf {
                row.msi[base + k] += svi * q.w * q.nm[k];
            }
            if logpart {
                row.ml[base + k] += psi * q.w * q.nm[k];
            }
        }
    }
}

// psi at the far points of block bk (instance ii): the rule of the wave part there (kernel_row), so that
// the subtraction takes out what the wave part holds
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn accumulate_far_psi(h: &HoGeom, im: &Images, ii: i32, bk: usize, row: &mut StaticRow) {
    let nk = h.bnk[bk] as usize;
    let NM = h.NM as usize;
    let col = &h.bcol[h.bcol0[bk] as usize..];
    let mut wo = h.f0[bk] as usize * nk; // offset of the weights of point qi in fwb
    for qi in h.f0[bk] as usize..h.f0[bk + 1] as usize {
        let wb = &h.fwb[wo..];
        wo += nk;
        let x = v3(h.fx[3 * qi], h.fx[3 * qi + 1], h.fx[3 * qi + 2]);
        let n = v3(h.fn_[3 * qi], h.fn_[3 * qi + 1], h.fn_[3 * qi + 2]);
        let dl = x - im.xb;
        let R = norm(dl);
        let den = R - dl[2];
        if !(den > 0.0) {
            continue;
        }
        let psi = den.ln();
        let dpsi = (dot(dl, n) / R - n[2]) / den;
        for t in 0..nk {
            row.ls[col[t] as usize] += psi * wb[t];
            row.ld[col[t] as usize] += dpsi * wb[t];
        }
        for m in 0..NM {
            row.ml[ii as usize * NM + m] += psi * h.fm[qi * NM + m];
        }
    }
}

// whether kernel_row integrates block bk at the target xt by the far rule: beyond WAVE_FAR
// block sizes from the image of xt in the free surface (and in finite depth in the bottom),
// where the regular part varies fastest. kernel_row and the psi subtraction of static_row both
// call it, so that the two choose the same rule.
pub(crate) fn wave_far(h: &HoGeom, bk: usize, xt: &Vec3) -> bool {
    let c = &h.bc[bk];
    let rx = xt[0] - c[0];
    let ry = xt[1] - c[1];
    let mut rz = (xt[2] + c[2]).abs();
    if h.depth.is_finite() {
        rz = rz.min((xt[2] + c[2] + 2.0 * h.depth).abs());
    }
    (rx * rx + ry * ry + rz * rz) / (h.bs[bk] * h.bs[bk]) > WAVE_FAR * WAVE_FAR
}

#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn gauss_rect(smp: &mut Sampler, ii: i32, iu: i32, iv: i32, u0: f64, u1: f64, v0: f64, v1: f64, nq: i32) {
    let x = gauss_nodes(nq, false);
    let w = gauss_nodes(nq, true);
    let hu = 0.5 * (u1 - u0);
    let hv = 0.5 * (v1 - v0);
    for a in 0..nq as usize {
        for b in 0..nq as usize {
            smp.sample(ii, iu, iv, u0 + hu * (x[a] + 1.0), v0 + hv * (x[b] + 1.0), w[a] * w[b] * hu * hv);
        }
    }
}

/// Parametric size of a rectangle of instance ii and its centre xc: the largest distance from
/// its centre to a corner.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn rect_size(h: &HoGeom, ii: i32, u0: f64, u1: f64, v0: f64, v1: f64, xc: &mut Vec3) -> f64 {
    let mut J = 0.0;
    let mut n = Vec3::ZERO;
    let mut c = [Vec3::ZERO; 4];
    let in_ = &h.inst[ii as usize];
    inst_eval(h, in_, 0.5 * (u0 + u1), 0.5 * (v0 + v1), xc, &mut n, &mut J, None, None);
    inst_eval(h, in_, u0, v0, &mut c[0], &mut n, &mut J, None, None);
    inst_eval(h, in_, u1, v0, &mut c[1], &mut n, &mut J, None, None);
    inst_eval(h, in_, u1, v1, &mut c[2], &mut n, &mut J, None, None);
    inst_eval(h, in_, u0, v1, &mut c[3], &mut n, &mut J, None, None);
    let mut s = 0.0f64;
    for q in &c {
        s = s.max(norm(*q - *xc));
    }
    s
}

// the physical lengths per unit u and v of a parametric rectangle (its mid-lines)
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn panel_lengths(h: &HoGeom, ii: i32, u0: f64, u1: f64, v0: f64, v1: f64, lu: &mut f64, lv: &mut f64) {
    let uc = 0.5 * (u0 + u1);
    let vc = 0.5 * (v0 + v1);
    let at = |u: f64, v: f64| -> Vec3 {
        let (mut x, mut n) = (Vec3::ZERO, Vec3::ZERO);
        let mut J = 0.0;
        inst_eval(h, &h.inst[ii as usize], u, v, &mut x, &mut n, &mut J, None, None);
        x
    };
    *lu = (norm(at(u1, vc) - at(u0, vc)) / (u1 - u0)).max(1e-300);
    *lv = (norm(at(uc, v1) - at(uc, v0)) / (v1 - v0)).max(1e-300);
}

#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn dmin_of(im: &Images, xc: &Vec3) -> f64 {
    let mut d = 1e300f64;
    for q in 0..im.n {
        d = d.min(norm(im.x[q] - *xc));
    }
    d.min(norm(im.xb - *xc))
}

// The images of im within lim of xc (near) and the others (far), the latter's least distance:
// the far ones are integrated by the rule of their own distance
fn split_images(im: &Images, xc: &Vec3, lim: f64, near: &mut Images, far: &mut Images, dfar: &mut f64) {
    near.n = 0;
    far.n = 0;
    near.xb = im.xb;
    far.xb = im.xb;
    *dfar = 1e300;
    for q in 0..im.n {
        let dq = norm(im.x[q] - *xc);
        let t = if dq <= lim { &mut *near } else { &mut *far };
        t.x[t.n] = im.x[q];
        t.si[t.n] = im.si[q];
        t.n += 1;
        if dq > lim {
            *dfar = dfar.min(dq);
        }
    }
}

// adaptive below NEAR_LEAF sizes, a leaf (6 x 6 Gauss) once its distance exceeds NEAR_LEAF
// sizes (Gauss-Legendre error of 1/r of the order rho^-12, 2e-8 on the panel normal, 6e-8 in its
// plane: theory §7.1; the leaves stop at 1.5 sizes rather than 3, which cut the static phase of
// an air-cushion vessel from 8.4 to 3.5 s, theory §7.1 grounds); between NEAR_LEAF and MID
// sizes the panel is such a leaf, its stored 6 x 6 points
const NEAR_LEAF: f64 = 1.5;
const MID: f64 = 6.0;
// beyond STATIC_FAR sizes from the target and its images the 3 x 3 rule: the nearest
// singularity lies 17 half-widths of the panel away, rho = 34, error rho^-6 < 1e-9 of the source
// kernel (the dipole 2.4e-8)
const STATIC_FAR: f64 = 12.0;
// the 3 x 3 rule integrates a polynomial of degree 5 per direction exactly: the basis functions of
// order k (degree k - 1) times the area element of a flat patch (degree 1) times the linear
// variation of the kernel across the panel (degree 1, its first Taylor term about the panel
// centre) has degree k + 1, which fits for k <= 4, so that the error is that of the kernel's
// higher terms (the rho^-6 above); higher orders keep the inner rule there
#[inline]
fn far_tier(d: &PatchDisc) -> bool {
    d.ku <= 4 && d.kv <= 4
}

#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn adapt(smp: &mut Sampler, ii: i32, iu: i32, iv: i32, u0: f64, u1: f64, v0: f64, v1: f64, depth: i32) {
    let mut xc = Vec3::ZERO;
    let s = rect_size(smp.h, ii, u0, u1, v0, v1, &mut xc);
    // depth 12: a cell 2^-12 of the panel, a guard against a singularity on the patch itself
    if depth >= 12 || s * NEAR_LEAF < dmin_of(smp.im, &xc) {
        gauss_rect(smp, ii, iu, iv, u0, u1, v0, v1, 6);
        return;
    }
    let um = 0.5 * (u0 + u1);
    let vm = 0.5 * (v0 + v1);
    adapt(smp, ii, iu, iv, u0, um, v0, vm, depth + 1);
    adapt(smp, ii, iu, iv, um, u1, v0, vm, depth + 1);
    adapt(smp, ii, iu, iv, um, u1, vm, v1, depth + 1);
    adapt(smp, ii, iu, iv, u0, um, vm, v1, depth + 1);
}

// Duffy transformation of a rectangle with the singular point at its corner
// (uc, vc) = (u0, v0), extents a = u1 - u0, b = v1 - v0 (either sign): two
// triangles, u = uc + a p, v = vc + b p q and v = vc + b p, u = uc + a p q.
// lu, lv: the physical lengths per unit u and v. In a triangle whose leg along q
// is rho times longer than its leg along p, 1/r varies in q on the scale 1/rho
// near q = 0: the q range is graded at 1/rho, 2/rho, 4/rho, ... (n points per
// piece, tested by the axisymmetry next to a pole).
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn duffy_rect(smp: &mut Sampler, ii: i32, iu: i32, iv: i32, uc: f64, vc: f64, a: f64, b: f64, lu: f64, lv: f64) {
    if a == 0.0 || b == 0.0 {
        return;
    }
    const N: usize = 8;
    let x = gauss_nodes(N as i32, false);
    let w = gauss_nodes(N as i32, true);
    let ab = (a * b).abs();
    let la = a.abs() * lu;
    let lb = b.abs() * lv;
    // a target in z = 0 with the log term: psi ~ log p after the map, so the p-integrand is
    // A(p) log p + B(p) (A, B smooth; the factor p of the map included), integrated by the
    // generalized Gauss rule exact for p^k and p^k log p (12 points; 8 Gauss points left 2e-4 of
    // p log p); in z = 0 within 1e-9 of the size of the rectangle (rounding of the map, as
    // stencil_top)
    let log_p = smp.logpart && smp.im.xb[2].abs() <= 1e-9 * la.max(lb);
    let np = if log_p { LOG_GAUSS_N } else { N };
    for t in 0..2 {
        let rho = if t == 0 { lb / la } else { la / lb };
        let mut q0 = 0.0;
        while q0 < 1.0 {
            let q1 = if rho > 2.0 { 1.0f64.min(if q0 == 0.0 { 1.0 / rho } else { 2.0 * q0 }) } else { 1.0 };
            let hq = q1 - q0;
            for i in 0..np {
                let (p, wp) =
                    if log_p { (LOG_GAUSS_NODES[i], LOG_GAUSS_WEIGHTS[i]) } else { (0.5 * (x[i] + 1.0), 0.5 * w[i]) };
                for j in 0..N {
                    let q = q0 + hq * 0.5 * (x[j] + 1.0);
                    let wq = 0.5 * hq * w[j];
                    if t == 0 {
                        smp.sample(ii, iu, iv, uc + a * p, vc + b * p * q, wp * wq * ab * p);
                    } else {
                        smp.sample(ii, iu, iv, uc + a * p * q, vc + b * p, wp * wq * ab * p);
                    }
                }
            }
            q0 = q1;
        }
    }
}

/// The static row of target xt (on instance tinst at (tu, tv), an outer point, or -1: off the
/// surfaces): all static integrals of a target point.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn static_row(
    h: &HoGeom,
    xt: &Vec3,
    tinst: i32,
    tu: f64,
    tv: f64,
    logint: bool,
    want_inf: bool,
    row: &mut StaticRow,
) {
    let ncol = h.Ncol as usize;
    let nim = h.inst.len() * h.NM as usize;
    row.rs = vec![0.0; ncol];
    row.rd = row.rs.clone();
    row.ls = row.rs.clone();
    row.ld = row.rs.clone();
    row.rsi = vec![0.0; if want_inf { ncol } else { 0 }];
    row.rdi = row.rsi.clone();
    row.ms = vec![0.0; nim];
    row.ml = row.ms.clone();
    row.msi = vec![0.0; if want_inf { nim } else { 0 }];
    let im = images_of(xt, h.depth);
    let mut tmp = StaticRow::default(); // the inner-rule value of the log part, subtracted below
    let (mo2, fo2) = ((MID_ORDER * MID_ORDER) as usize, (FAR_ORDER * FAR_ORDER) as usize);
    for ii in 0..h.inst.len() {
        let iii = ii as i32;
        let p = h.inst[ii].patch;
        let d = &h.disc[p as usize];
        for iv in 0..d.nv {
            for iu in 0..d.nu {
                let u0 = -1.0 + 2.0 * iu as f64 / d.nu as f64;
                let u1 = u0 + 2.0 / d.nu as f64;
                let v0 = -1.0 + 2.0 * iv as f64 / d.nv as f64;
                let v1 = v0 + 2.0 / d.nv as f64;
                let bk = block_of(h, iii, iu, iv) as usize;
                let xc = &h.bc[bk];
                let s = h.bs[bk];
                let dm = dmin_of(&im, xc);
                let on = tinst == iii && tu >= u0 && tu <= u1 && tv >= v0 && tv <= v1;
                // the log singularity is integrated separately wherever the inner Gauss
                // rule is not accurate, the near tier of the Rankine part
                let mut lg = false;
                if logint {
                    lg = s * MID >= dm;
                    if on {
                        lg = true; // the panel of a target in z = 0 (both on the free surface)
                    }
                }
                let far_pts = &h.sfar[bk * fo2..bk * fo2 + fo2];
                if !on && !lg && s * STATIC_FAR < dm && far_tier(d) {
                    accumulate_points(h, &im, iii, far_pts, want_inf, false, row);
                    continue;
                }
                // the middle tier (MID to STATIC_FAR sizes): the inner rule of the panel, or its 6 x 6
                // points when an inner order below 4 would leave it less accurate than the 3 x 3 tier beyond
                let inner_ok = d.iqui >= 4;
                let mpts: &[Pt] = if inner_ok {
                    &h.inner[h.inner0[bk] as usize..h.inner0[bk + 1] as usize]
                } else {
                    &h.mid[bk * mo2..bk * mo2 + mo2]
                };
                let tier_inner = !on && s * MID < dm;
                if tier_inner && !lg {
                    accumulate_points(h, &im, iii, mpts, want_inf, false, row);
                    continue;
                }
                // the images beyond MID sizes by the inner or far rule, the near ones below
                let (mut nim_, mut fim) = (Images::default(), Images::default());
                let mut dfar = 0.0;
                split_images(&im, xc, s * MID, &mut nim_, &mut fim, &mut dfar);
                if fim.n > 0 {
                    let far = s * STATIC_FAR < dfar && far_tier(d);
                    let pts = if far { far_pts } else { mpts };
                    accumulate_points(h, &fim, iii, pts, want_inf, false, row);
                }
                {
                    let mut smp = Sampler { h, im: &nim_, row: &mut *row, want_inf, logpart: lg };
                    if on {
                        let (mut lu, mut lv) = (0.0, 0.0);
                        panel_lengths(h, iii, u0, u1, v0, v1, &mut lu, &mut lv);
                        duffy_rect(&mut smp, iii, iu, iv, tu, tv, u1 - tu, v1 - tv, lu, lv);
                        duffy_rect(&mut smp, iii, iu, iv, tu, tv, u0 - tu, v1 - tv, lu, lv);
                        duffy_rect(&mut smp, iii, iu, iv, tu, tv, u0 - tu, v0 - tv, lu, lv);
                        duffy_rect(&mut smp, iii, iu, iv, tu, tv, u1 - tu, v0 - tv, lu, lv);
                    } else if s * NEAR_LEAF >= dm {
                        adapt(&mut smp, iii, iu, iv, u0, u1, v0, v1, 0);
                    } else {
                        accumulate_points(h, &nim_, iii, &h.mid[bk * mo2..bk * mo2 + mo2], want_inf, lg, smp.row);
                    }
                }
                if lg {
                    // minus the value of psi that the wave part holds: by its inner rule, or by its far rule
                    // beyond WAVE_FAR sizes
                    if tmp.ls.is_empty() {
                        tmp.rs = vec![0.0; ncol];
                        tmp.rd = tmp.rs.clone();
                        tmp.ls = tmp.rs.clone();
                        tmp.ld = tmp.rs.clone();
                        tmp.ms = vec![0.0; nim];
                        tmp.ml = tmp.ms.clone();
                    }
                    if wave_far(h, bk, xt) {
                        accumulate_far_psi(h, &im, iii, bk, &mut tmp);
                    } else {
                        accumulate_points(
                            h,
                            &im,
                            iii,
                            &h.inner[h.inner0[bk] as usize..h.inner0[bk + 1] as usize],
                            false,
                            true,
                            &mut tmp,
                        );
                    }
                }
            }
        }
    }
    if !tmp.ls.is_empty() {
        for c in 0..ncol {
            row.ls[c] -= tmp.ls[c];
            row.ld[c] -= tmp.ld[c];
        }
        for k in 0..row.ml.len() {
            row.ml[k] -= tmp.ml[k];
        }
    }
    row.compress();
}

impl StaticRow {
    /// The log corrections kept sparse: ls, ld and ml emptied.
    pub fn compress(&mut self) {
        // the sparse form from the dense one alone: a row compressed again (or reused) keeps no old entries
        self.lcol.clear();
        self.lsv.clear();
        self.ldv.clear();
        self.mcol.clear();
        self.mlv.clear();
        for c in 0..self.ls.len() {
            if self.ls[c] != 0.0 || self.ld[c] != 0.0 {
                self.lcol.push(c as i32);
                self.lsv.push(self.ls[c]);
                self.ldv.push(self.ld[c]);
            }
        }
        for k in 0..self.ml.len() {
            if self.ml[k] != 0.0 {
                self.mcol.push(k as i32);
                self.mlv.push(self.ml[k]);
            }
        }
        self.ls = Vec::new();
        self.ld = Vec::new();
        self.ml = Vec::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_keeps_the_nonzero_entries() {
        let mut r = StaticRow {
            ls: vec![0.0, 1.5, 0.0, 0.0],
            ld: vec![0.0, 0.0, -2.0, 0.0],
            ml: vec![0.0, 3.0],
            ..StaticRow::default()
        };
        r.lcol = vec![7]; // an old sparse entry is dropped
        r.compress();
        assert_eq!(r.lcol, vec![1, 2]);
        assert_eq!(r.lsv, vec![1.5, 0.0]);
        assert_eq!(r.ldv, vec![0.0, -2.0]);
        assert_eq!(r.mcol, vec![1]);
        assert_eq!(r.mlv, vec![3.0]);
        assert!(r.ls.is_empty() && r.ld.is_empty() && r.ml.is_empty());
        // compressed again: nothing left (the dense form is empty)
        r.compress();
        assert!(r.lcol.is_empty() && r.mcol.is_empty());
    }

    #[test]
    fn the_log_gauss_rule_is_exact_for_its_functions() {
        // p^k and p^k log p, k = 0..11: 1/(k+1) and -1/(k+1)^2
        for k in 0..LOG_GAUSS_N as i32 {
            let (mut a, mut b) = (0.0, 0.0);
            for i in 0..LOG_GAUSS_N {
                let (p, w) = (LOG_GAUSS_NODES[i], LOG_GAUSS_WEIGHTS[i]);
                a += w * p.powi(k);
                b += w * p.powi(k) * p.ln();
            }
            let k1 = (k + 1) as f64;
            assert!((a - 1.0 / k1).abs() < 1e-15 && (b + 1.0 / (k1 * k1)).abs() < 1e-15, "{k}: {a} {b}");
        }
    }

    #[test]
    fn images_and_split() {
        let x = v3(1.0, 2.0, -3.0);
        let im = images_of(&x, f64::INFINITY);
        assert_eq!(im.n, 2);
        assert_eq!(im.x[1], v3(1.0, 2.0, 3.0));
        assert_eq!((im.si[0], im.si[1]), (1.0, -1.0));
        let im = images_of(&x, 10.0);
        assert_eq!(im.n, 6);
        assert_eq!(im.x[1], v3(1.0, 2.0, -17.0));
        assert_eq!(im.x[5], v3(1.0, 2.0, -37.0));
        assert_eq!(im.si, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let (mut n, mut f) = (Images::default(), Images::default());
        let mut dfar = 0.0;
        split_images(&im, &v3(1.0, 2.0, -2.0), 6.0, &mut n, &mut f, &mut dfar);
        // near: x (1 away) and -x (5 away); far: the bottom images
        assert_eq!((n.n, f.n), (2, 4));
        assert_eq!(n.x[1], v3(1.0, 2.0, 3.0));
        assert_eq!(dfar, 15.0);
        assert_eq!(dmin_of(&im, &v3(1.0, 2.0, -2.0)), 1.0);
    }
}
