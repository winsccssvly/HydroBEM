//! Higher-order method: one frequency: assembly, solve, forces,
//! outputs and the pressure drift (theory §7, theory §2.1, §4.1, §11.2). The spline values and the
//! L2 projection are in ho_internal.rs, the compact point arrays and the projection's factors in
//! ho_geometry.rs. The systems stay complex128 (theory §7, grounds of
//! the precision: condition numbers 2e2-3e3 and the body velocity a derivative of the solution);
//! the reduced far rule of the wave part and the four-point kernels: theory §7.2, §8.3.

use crate::codes::Stage;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

use crate::constants::PI;
use crate::control_drift::ControlSurfaceDrift;
use crate::error::{Error, Result};
use crate::geometry::{Vec3, v3};
use crate::green_fin::{FinPoint, FiniteDepth, FreqMode, Kern, fin_point, fin4_applies, finite_kernel4};
use crate::green_inf::{direct4_applies, near4_applies, wave_kernel_direct4, wave_kernel_near4};
use crate::ho_geometry::point_modes;
use crate::ho_geometry::stencil_top;
use crate::ho_internal::{
    BlockRule, HoGeom, MAX_MODES, MAX_PANEL_BASIS, Pt, SheetEval, SheetRowData, StaticRow, inst_tangents, mode_index,
    mode_sign, project, spline_values, surface_gradient,
};
use crate::ho_sheets::sheet_wave_row;
use crate::ho_static::wave_far;
use crate::linalg::lu_solve_c128;
use crate::low_order::{
    FreqCtx, Geometry, Output, Plan, Profile, SurfaceKind, channel_far, far_kernel, incident, kochin_global,
    pair_kernel, wave_kern,
};
use crate::motion::solve_motions;
use crate::parallel::try_for_each_grouped;
use crate::pressure_drift::PressureDrift;
use crate::problem::Problem;
use crate::progress::Work;
use crate::quadratic_qtf::{CsRecorder, QtfRecorder};
use crate::tanks::add_tank_terms;

const Z: Complex64 = Complex64::new(0.0, 0.0);

/// z rounded to complex float32 (each part rounded).
#[inline]
fn to_cf(z: Complex64) -> Complex32 {
    Complex32::new(z.re as f32, z.im as f32)
}

/// z widened to complex float64.
#[inline]
fn to_cd(z: Complex32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}

// The per-frequency depth factors of the inner and far points (finite depth; empty otherwise).
#[derive(Default)]
struct FinPoints {
    in_: Vec<FinPoint>,
    far: Vec<FinPoint>,
}

fn fin_points(h: &HoGeom, fc: &FreqCtx) -> FinPoints {
    let mut f = FinPoints::default();
    if !fc.finite() {
        return f;
    }
    for q in 0..h.wx.len() / 3 {
        f.in_.push(fin_point(&fc.fd, h.wx[3 * q + 2]));
    }
    for q in 0..h.fx.len() / 3 {
        f.far.push(fin_point(&fc.fd, h.fx[3 * q + 2]));
    }
    f
}

/// Which kernel_row_near4 serves the rows: Some(false) at ordinary frequencies in infinite depth,
/// Some(true) in finite depth with the depth factors of the points (fin) and no channel; None:
/// kernel_row.
#[inline]
fn near4_mode(fc: &FreqCtx, h: &HoGeom, fin: &FinPoints) -> Option<bool> {
    if !fc.finite() {
        (fc.mode == FreqMode::Ordinary).then_some(false)
    } else {
        (h.W == 0.0 && !fin.in_.is_empty()).then_some(true)
    }
}

/// The kernels of the target xt and four consecutive points x (12 coordinates) by
/// wave_kernel_near4 or wave_kernel_direct4, as pair_kernel gives them, when the four are in one
/// tile of the near branch or of the direct tables; false (kb untouched) otherwise.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn near4_kernels(K: f64, xt: &Vec3, x: &[f64], kb: &mut [(Kern, f64); 4]) -> bool {
    let R: [f64; 4] = std::array::from_fn(|k| {
        let dx = xt[0] - x[3 * k];
        let dy = xt[1] - x[3 * k + 1];
        (dx * dx + dy * dy).sqrt()
    });
    let Xs: [f64; 4] = std::array::from_fn(|k| K * R[k]);
    let Ys: [f64; 4] = std::array::from_fn(|k| -K * (xt[2] + x[3 * k + 2]));
    let w = if near4_applies(&Xs, &Ys) {
        wave_kernel_near4(&Xs, &Ys)
    } else if direct4_applies(&Xs, &Ys) {
        wave_kernel_direct4(&Xs, &Ys)
    } else {
        return false;
    };
    for k in 0..4 {
        kb[k] = (wave_kern(K, &w[k]), R[k]);
    }
    true
}

/// The finite-depth kernels of the target xt (depth factors ft) and four consecutive points x (12
/// coordinates, depth factors f) by finite_kernel4, as pair_kernel gives them with the depth
/// factors, when the four are in the tiled integral form; false (kb untouched) otherwise.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn fin4_kernels(
    fd: &FiniteDepth,
    xt: &Vec3,
    ft: &FinPoint,
    x: &[f64],
    f: &[FinPoint],
    kb: &mut [(Kern, f64); 4],
) -> bool {
    let R: [f64; 4] = std::array::from_fn(|k| {
        let dx = xt[0] - x[3 * k];
        let dy = xt[1] - x[3 * k + 1];
        (dx * dx + dy * dy).sqrt()
    });
    if !fin4_applies(fd, &R) {
        return false;
    }
    let zeta: [f64; 4] = std::array::from_fn(|k| x[3 * k + 2]);
    let w = finite_kernel4(fd, &R, xt[2], &zeta, ft, std::array::from_fn(|k| &f[k]));
    for k in 0..4 {
        kb[k] = (w[k], R[k]);
    }
    true
}

/// The static part of the row of a target point (the columns S, D sized to the columns, M to the
/// instances and modes): the regular part of its StaticRow, then the sparse log corrections
/// (StaticRow::compress), of the ordinary or the infinite frequency; whether a wave part follows
/// (finite depth, or an ordinary frequency).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn static_columns(
    h: &HoGeom,
    fc: &FreqCtx,
    st: &StaticRow,
    want_s: bool,
    S: &mut Vec<Complex64>,
    D: &mut Vec<Complex64>,
    M: &mut Vec<Complex64>,
) -> bool {
    let inf = fc.mode == FreqMode::Infinite;
    let Ncol = h.Ncol as usize;
    S.clear();
    S.resize(if want_s { Ncol } else { 0 }, Z);
    D.clear();
    D.resize(Ncol, Z);
    M.clear();
    M.resize(h.inst.len() * h.NM as usize, Z);
    let c2 = 2.0 * fc.K * fc.logc;
    // the regular part, then the sparse log corrections (StaticRow::compress)
    for c in 0..Ncol {
        if want_s {
            S[c] = Complex64::new(if inf { st.rsi[c] } else { st.rs[c] }, 0.0);
        }
        D[c] = Complex64::new(if inf { st.rdi[c] } else { st.rd[c] }, 0.0);
    }
    for t in 0..st.lcol.len() {
        let c = st.lcol[t] as usize;
        if want_s {
            S[c] = Complex64::new((if inf { st.rsi[c] } else { st.rs[c] }) - c2 * st.lsv[t], 0.0);
        }
        D[c] = Complex64::new((if inf { st.rdi[c] } else { st.rd[c] }) - c2 * st.ldv[t], 0.0);
    }
    for k in 0..M.len() {
        M[k] = Complex64::new(if inf { st.msi[k] } else { st.ms[k] }, 0.0);
    }
    for t in 0..st.mcol.len() {
        let k = st.mcol[t] as usize;
        M[k] = Complex64::new((if inf { st.msi[k] } else { st.ms[k] }) - c2 * st.mlv[t], 0.0);
    }
    fc.finite() || fc.mode == FreqMode::Ordinary
}

/// Wave (regular) part of a target point against all panels, plus the static parts
/// (static_columns): per column the source and dipole, per (instance, mode) the source. Finite
/// depth: fin the per-frequency depth factors of the inner and far points (fin_points).
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn kernel_row(
    h: &HoGeom,
    fc: &FreqCtx,
    fin: &FinPoints,
    xt: &Vec3,
    st: &StaticRow,
    want_s: bool,
    S: &mut Vec<Complex64>,
    D: &mut Vec<Complex64>,
    M: &mut Vec<Complex64>,
) {
    if !static_columns(h, fc, st, want_s, S, D, M) {
        return;
    }
    // finite depth: the depth factors of the target once per row, of the points once per frequency
    let fp = fc.finite() && !fin.in_.is_empty();
    let ft = if fp { fin_point(&fc.fd, xt[2]) } else { FinPoint::default() };
    // per (instance, panel) block: sums over its points, then one scatter
    let NM = h.NM as usize;
    let nb = h.inner0.len() - 1;
    let mut dre = [0.0f64; MAX_PANEL_BASIS];
    let mut dim = [0.0f64; MAX_PANEL_BASIS];
    let mut sre = [0.0f64; MAX_PANEL_BASIS];
    let mut sim = [0.0f64; MAX_PANEL_BASIS];
    let mut mre = [0.0f64; MAX_MODES];
    let mut mim = [0.0f64; MAX_MODES];
    for bk in 0..nb {
        let a0 = h.inner0[bk] as usize;
        let e0 = h.inner0[bk + 1] as usize;
        if a0 == e0 {
            continue;
        }
        let nk = h.bnk[bk] as usize;
        for t in 0..nk {
            dre[t] = 0.0;
            dim[t] = 0.0;
            sre[t] = 0.0;
            sim[t] = 0.0;
        }
        for m in 0..NM {
            mre[m] = 0.0;
            mim[m] = 0.0;
        }
        // far from the target and its free-surface image: the lower-order rule
        let far = wave_far(h, bk, xt);
        let BlockRule { q0, q1, wb: wbv, wb0: mut wbo, x: X, n: Nn, m: Wm } = h.block_rule(bk, far);
        let fq: Option<&[FinPoint]> = if fp { Some(if far { &fin.far } else { &fin.in_ }) } else { None };
        for qi in q0..q1 {
            let x = v3(X[3 * qi], X[3 * qi + 1], X[3 * qi + 2]);
            let n = &Nn[3 * qi..3 * qi + 3];
            let (k, R) = pair_kernel(fc, xt, &x, fq.map(|_| &ft), fq.map(|f| &f[qi]));
            let a = if R > 0.0 { ((xt[0] - x[0]) * n[0] + (xt[1] - x[1]) * n[1]) / R } else { 0.0 };
            let dr = -k.WrR * a + k.WrZ * n[2];
            let di = -k.WiR * a + k.WiZ * n[2];
            let wb = &wbv[wbo..wbo + nk];
            for t in 0..nk {
                dre[t] += dr * wb[t];
                dim[t] += di * wb[t];
            }
            if want_s {
                for t in 0..nk {
                    sre[t] += k.Wr * wb[t];
                    sim[t] += k.Wi * wb[t];
                }
            }
            let wm = &Wm[qi * NM..qi * NM + NM];
            for m in 0..NM {
                mre[m] += k.Wr * wm[m];
                mim[m] += k.Wi * wm[m];
            }
            wbo += nk;
        }
        let col = &h.bcol[h.bcol0[bk] as usize..];
        for t in 0..nk {
            D[col[t] as usize] += Complex64::new(dre[t], dim[t]);
        }
        if want_s {
            for t in 0..nk {
                S[col[t] as usize] += Complex64::new(sre[t], sim[t]);
            }
        }
        let mq = h.inner[a0].inst as usize * NM;
        for m in 0..NM {
            M[mq + m] += Complex64::new(mre[m], mim[m]);
        }
    }
    if h.W > 0.0 {
        channel_terms(h, fc, fin, xt, &ft, want_s, S, D, M);
    }
}

/// kernel_row at an ordinary frequency in infinite depth (FIN false), with the kernels of four
/// points at a time by wave_kernel_near4 where they share a table tile, or in finite depth without
/// a channel (FIN true), by finite_kernel4 where the four are in the tiled integral form: the same
/// sums in the same order, equal to kernel_row's to the bit. kernel_row serves the other modes (the
/// limits, the channel).
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn kernel_row_near4<const FIN: bool>(
    h: &HoGeom,
    fc: &FreqCtx,
    fin: &FinPoints,
    xt: &Vec3,
    st: &StaticRow,
    want_s: bool,
    S: &mut Vec<Complex64>,
    D: &mut Vec<Complex64>,
    M: &mut Vec<Complex64>,
) {
    if !static_columns(h, fc, st, want_s, S, D, M) {
        return;
    }
    // finite depth: the depth factors of the target once per row, of the points once per frequency
    let fp = fc.finite() && !fin.in_.is_empty();
    let ft = if fp { fin_point(&fc.fd, xt[2]) } else { FinPoint::default() };
    // per (instance, panel) block: sums over its points, then one scatter
    let NM = h.NM as usize;
    let nb = h.inner0.len() - 1;
    let mut dre = [0.0f64; MAX_PANEL_BASIS];
    let mut dim = [0.0f64; MAX_PANEL_BASIS];
    let mut sre = [0.0f64; MAX_PANEL_BASIS];
    let mut sim = [0.0f64; MAX_PANEL_BASIS];
    let mut mre = [0.0f64; MAX_MODES];
    let mut mim = [0.0f64; MAX_MODES];
    let mut kb = [(Kern::default(), 0.0f64); 4];
    for bk in 0..nb {
        let a0 = h.inner0[bk] as usize;
        let e0 = h.inner0[bk + 1] as usize;
        if a0 == e0 {
            continue;
        }
        let nk = h.bnk[bk] as usize;
        for t in 0..nk {
            dre[t] = 0.0;
            dim[t] = 0.0;
            sre[t] = 0.0;
            sim[t] = 0.0;
        }
        for m in 0..NM {
            mre[m] = 0.0;
            mim[m] = 0.0;
        }
        // far from the target and its free-surface image: the lower-order rule
        let far = wave_far(h, bk, xt);
        let BlockRule { q0, q1, wb: wbv, wb0: mut wbo, x: X, n: Nn, m: Wm } = h.block_rule(bk, far);
        let fq: Option<&[FinPoint]> = if fp { Some(if far { &fin.far } else { &fin.in_ }) } else { None };
        // the kernels four points at a time (infinite depth: wave_kernel_near4 where the four
        // share a table tile; finite depth: finite_kernel4 where they are in the tiled integral
        // form; else one by one), then the sums in the order of the points
        let mut qn = q0;
        while qn < q1 {
            let nq = (q1 - qn).min(4);
            let four = nq == 4
                && if FIN {
                    fq.is_some_and(|f| fin4_kernels(&fc.fd, xt, &ft, &X[3 * qn..3 * qn + 12], &f[qn..qn + 4], &mut kb))
                } else {
                    near4_kernels(fc.K, xt, &X[3 * qn..3 * qn + 12], &mut kb)
                };
            if !four {
                for j in 0..nq {
                    let qi = qn + j;
                    let x = v3(X[3 * qi], X[3 * qi + 1], X[3 * qi + 2]);
                    kb[j] = pair_kernel(fc, xt, &x, fq.map(|_| &ft), fq.map(|f| &f[qi]));
                }
            }
            for (j, &(k, R)) in kb[..nq].iter().enumerate() {
                let qi = qn + j;
                let x = v3(X[3 * qi], X[3 * qi + 1], X[3 * qi + 2]);
                let n = &Nn[3 * qi..3 * qi + 3];
                let a = if R > 0.0 { ((xt[0] - x[0]) * n[0] + (xt[1] - x[1]) * n[1]) / R } else { 0.0 };
                let dr = -k.WrR * a + k.WrZ * n[2];
                let di = -k.WiR * a + k.WiZ * n[2];
                let wb = &wbv[wbo..wbo + nk];
                for t in 0..nk {
                    dre[t] += dr * wb[t];
                    dim[t] += di * wb[t];
                }
                if want_s {
                    for t in 0..nk {
                        sre[t] += k.Wr * wb[t];
                        sim[t] += k.Wi * wb[t];
                    }
                }
                let wm = &Wm[qi * NM..qi * NM + NM];
                for m in 0..NM {
                    mre[m] += k.Wr * wm[m];
                    mim[m] += k.Wi * wm[m];
                }
                wbo += nk;
            }
            qn += nq;
        }
        let col = &h.bcol[h.bcol0[bk] as usize..];
        for t in 0..nk {
            D[col[t] as usize] += Complex64::new(dre[t], dim[t]);
        }
        if want_s {
            for t in 0..nk {
                S[col[t] as usize] += Complex64::new(sre[t], sim[t]);
            }
        }
        let mq = h.inner[a0].inst as usize * NM;
        for m in 0..NM {
            M[mq + m] += Complex64::new(mre[m], mim[m]);
        }
    }
    debug_assert!(h.W == 0.0, "a channel is in finite depth");
}

/// The channel terms of a wave row (theory §8.5): the near images (instances) scaled by the
/// reflection factor, and the farther images of every original instance (channel_far) at its
/// points: the far rule, or the inner rule within WAVE_FAR sizes of the nearest far image.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn channel_terms(
    h: &HoGeom,
    fc: &FreqCtx,
    fin: &FinPoints,
    xt: &Vec3,
    ft: &FinPoint,
    want_s: bool,
    S: &mut [Complex64],
    D: &mut [Complex64],
    M: &mut [Complex64],
) {
    let NM = h.NM as usize;
    let Rc = h.scale_wall_images(D, if want_s { Some(S) } else { None }, M);
    if Rc == 0.0 {
        return;
    }
    let W = h.W;
    let nb = h.inner0.len() - 1;
    for bk in 0..nb {
        let a0 = h.inner0[bk] as usize;
        let e0 = h.inner0[bk + 1] as usize;
        if a0 == e0 || h.inst[h.inner[a0].inst as usize].image {
            continue;
        }
        let nk = h.bnk[bk] as usize;
        let far = h.far_from_wall_images(bk, xt);
        let BlockRule { q0, q1, wb: wbv, wb0: mut wbo, x: X, n: Nn, m: Wm } = h.block_rule(bk, far);
        let fq = if far { &fin.far } else { &fin.in_ };
        let mut ds = [Z; MAX_PANEL_BASIS];
        let mut ss = [Z; MAX_PANEL_BASIS];
        let mut ms = [Z; MAX_MODES];
        for qi in q0..q1 {
            let x = v3(X[3 * qi], X[3 * qi + 1], X[3 * qi + 2]);
            let n = v3(Nn[3 * qi], Nn[3 * qi + 1], Nn[3 * qi + 2]);
            let (g, dg) = channel_far(fc, W, xt, &x, &n, 1.0, ft, &fq[qi]);
            let wb = &wbv[wbo..wbo + nk];
            for t in 0..nk {
                ds[t] += dg * wb[t];
                if want_s {
                    ss[t] += g * wb[t];
                }
            }
            let wm = &Wm[qi * NM..qi * NM + NM];
            for m in 0..NM {
                ms[m] += g * wm[m];
            }
            wbo += nk;
        }
        let col = &h.bcol[h.bcol0[bk] as usize..];
        for t in 0..nk {
            D[col[t] as usize] += Rc * ds[t];
            if want_s {
                S[col[t] as usize] += Rc * ss[t];
            }
        }
        let mq = h.inner[a0].inst as usize * NM;
        for m in 0..NM {
            M[mq + m] += Rc * ms[m];
        }
    }
}

// The data of the Galerkin rows of one outer point (body_rows): the dipole row of each character
// (NC x N), the free term, the radiation sums of the modes (nrad) and the diffraction right-hand
// sides of each character (NC x nh), without the test-function weight.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct RowData {
    skip: bool, // a sheet point (its rows are collocation rows)
    free_t: f64,
    Dc: Vec<Complex64>,
    vr: Vec<Complex64>,
    vh: Vec<Complex64>,
}

// The columns of the patch kinds other than the wetted surface, in one place for the four users.
// A kernel row gives per column the source S and dipole D integrals and per instance and mode the
// source integrals M of the normal velocity; the column of an instance becomes (column_rule):
//
//   kind                     Equation (body and sheet rows)   Field (field points, Kochin)
//   unused (unused())        0                                (the fields are not computed there)
//   interior free surface    K S (0 at the limits)            0, and its S and M are 0 too
//   damping lid              -(K - kappa) S (Robin)           -(K - kappa) S
//   free-surface plate       -K S (0 at omega = 0;            -K S
//                            D, M = 0 at omega = inf)
//   pressure surface, z = 0  0                                0
//   pressure surface         D + K S                          D + K S
//   sheet                    D; M = 0 (the body rows)         D; M = 0
//
// The interior free surface is part of the equations (irregular-frequency removal) but does not
// bound the fluid, so it leaves Green's identity in the fluid.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    Equation,
    Field,
}

// The dipole column of an instance from its source column.
#[derive(Clone, Copy)]
enum ColumnRule {
    Zero,
    Scaled(f64),      // a S
    Robin(Complex64), // a S, a lid: a = -(K - kappa)
    Added(f64),       // D + a S
}

// d[l] (the dipole column of an instance, its columns c0 ..) by the rule from s[c0 + l]
fn apply_rule(rule: ColumnRule, d: &mut [Complex64], s: &[Complex64], c0: usize) {
    for (l, dl) in d.iter_mut().enumerate() {
        *dl = match rule {
            ColumnRule::Zero => Z,
            ColumnRule::Scaled(a) => a * s[c0 + l],
            ColumnRule::Robin(a) => a * s[c0 + l],
            ColumnRule::Added(a) => *dl + a * s[c0 + l],
        };
    }
}

// The mode of radiation column `mode` is even under character c (the same sign under every
// element of the group).
fn matches(h: &HoGeom, c: usize, mode: i32) -> bool {
    (0..h.G).all(|g| h.chi[c][g as usize] == mode_sign(h, g, mode))
}

// The source, dipole and mode columns of one kernel row (kernel_row) or far-field row (kochin).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[derive(Default)]
struct Columns {
    S: Vec<Complex64>,
    D: Vec<Complex64>,
    M: Vec<Complex64>,
}

impl Default for RowData {
    fn default() -> Self {
        RowData { skip: true, free_t: 0.0, Dc: Vec::new(), vr: Vec::new(), vh: Vec::new() }
    }
}

// One frequency or limit of the higher-order method. The steps of solve_frequency (below) are
// its methods, in their order; they share the sizes, the incident wave, the solved
// B-spline coefficients and the field values. (The outputs are passed to the steps that write
// them.)
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct Frequency<'a> {
    m: &'a Problem,
    h: &'a HoGeom<'a>,
    pl: &'a Plan,
    se: &'a [SheetEval],
    so: &'a [StaticRow],
    sf: &'a [StaticRow],
    fc: &'a FreqCtx,
    work: Work<'a>,
    limit: bool,
    w: f64,
    K: f64,
    nrad: usize,
    nh: usize,
    NC: usize,
    N: usize,
    Ninst: usize,
    NM: usize,
    NQ: usize,
    fin: FinPoints, // finite depth: the depth factors of the points
    inner: i32,     // threads inside the frequency: the batches of rows, the characters solved at once
    fg: Geometry,   // the frame only, for the incident wave
    need_s: bool,   // the source columns are needed
    nrhs: usize,
    D6: usize,
    rho: f64,
    iwr: Complex64,
    A: Vec<Vec<Complex64>>, // [character] the Galerkin system, row-major until the solve
    B: Vec<Vec<Complex64>>,
    phiI: Vec<Complex64>, // [nh][NQ] the incident wave at the inner points
    dphiI: Vec<Complex64>,
    uS: Vec<Vec<Vec<Complex64>>>, // scattered: the projected boundary values
    uSi: Vec<Vec<Vec<Complex64>>>,
    rmode: Vec<i32>,           // [nrad] the mode index of each radiation column
    coef: Vec<Vec<Complex64>>, // [nrhs] the coefficients of every instance, values at the inner points
    phi_q: Vec<Vec<Complex64>>,
    Pall: usize, // field points (with the stencils), output points, control surface
    P: usize,
    NCs: i32,
    Gc: usize,
    fpr: Vec<Complex64>, // the potentials at the field points (and reflections)
    fps: Vec<Complex64>,
    fprg: Vec<Complex64>,
    fpsg: Vec<Complex64>,
}

impl<'a> Frequency<'a> {
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn new(
        m: &'a Problem,
        h: &'a HoGeom<'a>,
        pl: &'a Plan,
        se: &'a [SheetEval],
        so: &'a [StaticRow],
        sf: &'a [StaticRow],
        fc: &'a FreqCtx,
        work: Work<'a>,
    ) -> Frequency<'a> {
        let limit = fc.mode != FreqMode::Ordinary;
        let nrad = pl.rad.len();
        let nh = if pl.diffraction && !limit { pl.heads.len() } else { 0 };
        let NC = h.chi.len();
        let N = h.Ndof as usize;
        let fg = Geometry { yaw: h.yaw, px: h.px, py: h.py, walls: h.walls.clone(), ..Geometry::default() };
        let need_s = (!pl.total && nh > 0) || h.kind.iter().any(|&c| c != SurfaceKind::Wet);
        let nrhs = nrad + nh;
        Frequency {
            m,
            h,
            pl,
            se,
            so,
            sf,
            fc,
            work,
            limit,
            w: fc.w,
            K: fc.K,
            nrad,
            nh,
            NC,
            N,
            Ninst: h.inst.len(),
            NM: h.NM as usize,
            NQ: h.inner.len(),
            fin: fin_points(h, fc),
            inner: m.opt.inner_threads,
            fg,
            need_s,
            nrhs,
            D6: pl.D as usize,
            rho: pl.rho,
            iwr: Complex64::new(0.0, fc.w * pl.rho),
            A: vec![vec![Z; N * N]; NC],
            B: vec![vec![Z; N * nrhs.max(1)]; NC],
            phiI: Vec::new(),
            dphiI: Vec::new(),
            uS: Vec::new(),
            uSi: Vec::new(),
            rmode: Vec::new(),
            coef: Vec::new(),
            phi_q: Vec::new(),
            Pall: 0,
            P: 0,
            NCs: 0,
            Gc: 1,
            fpr: Vec::new(),
            fps: Vec::new(),
            fprg: Vec::new(),
            fpsg: Vec::new(),
        }
    }

    // incident wave (a geometry with the frame only): value and gradient
    fn inc(&self, hh: usize, x: &Vec3) -> (Complex64, [Complex64; 3]) {
        incident(&self.fg, self.pl, self.fc, self.pl.heads[hh], x)
    }
    // the free-surface patches of the system: interior surfaces are not used at the
    // limits, pressure surfaces in z = 0 have phi = 0 at omega = inf
    fn unused(&self, ii: i32) -> bool {
        let kd = self.h.kind_of(ii);
        (self.limit && (kd == SurfaceKind::Interior || kd == SurfaceKind::Lid))
            || (self.h.z0(ii) && self.fc.mode == FreqMode::Infinite)
    }
    // a sheet, or at omega = inf a free-surface plate: with the odd image of phi = 0 on the free
    // surface the plate is a sheet in z = 0 whose image doubles its dipole and hypersingular
    // kernels (theory §9.8); its unknown phi is that of the water side, half the jump
    fn sheet_like(&self, ii: i32) -> bool {
        let kd = self.h.kind_of(ii);
        kd == SurfaceKind::Sheet || (kd == SurfaceKind::Plate && self.fc.mode == FreqMode::Infinite)
    }
    // the rule of the dipole column of instance ii (not wetted, not a sheet; the table at Target)
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn column_rule(&self, ii: i32, target: Target) -> ColumnRule {
        let h = self.h;
        let kd = h.kind_of(ii);
        let K = self.K;
        let robin = || ColumnRule::Robin(-(K - self.fc.kappa[h.lid_of(ii) as usize]));
        // a free-surface plate: dG/dn = -K G on zeta = 0 (normal -z), the lid of kappa = 0; its
        // unknowns are used at omega = 0 too, where the column vanishes (theory §9.7)
        let plate = || if self.limit { ColumnRule::Zero } else { ColumnRule::Robin(Complex64::new(-K, 0.0)) };
        match target {
            Target::Equation => {
                if self.unused(ii) {
                    ColumnRule::Zero
                } else if kd == SurfaceKind::Interior {
                    if self.limit { ColumnRule::Zero } else { ColumnRule::Scaled(K) }
                } else if kd == SurfaceKind::Lid {
                    robin()
                } else if kd == SurfaceKind::Plate {
                    plate()
                } else if h.z0(ii) {
                    ColumnRule::Zero
                } else {
                    ColumnRule::Added(K)
                }
            }
            Target::Field => {
                if kd == SurfaceKind::Interior || h.z0(ii) {
                    ColumnRule::Zero
                } else if kd == SurfaceKind::Lid {
                    robin()
                } else if kd == SurfaceKind::Plate {
                    plate()
                } else {
                    ColumnRule::Added(K)
                }
            }
        }
    }
    // the columns of every instance of the other patch kinds than the wetted surface (the table at
    // Target): D from S, M of the sheets 0, and for the field those of the interior surfaces 0;
    // m None leaves the mode columns
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn columns(&self, target: Target, S: &mut [Complex64], D: &mut [Complex64], mut m: Option<&mut [Complex64]>) {
        let h = self.h;
        let NM = self.NM;
        for ii in 0..self.Ninst {
            let kd = h.kind_of(ii as i32);
            if kd == SurfaceKind::Wet {
                continue;
            }
            if self.sheet_like(ii as i32) {
                // dipoles only (a plate at omega = inf: its single layer of G = 1/r - 1/r' vanishes)
                if let Some(m) = m.as_deref_mut() {
                    m[ii * NM..(ii + 1) * NM].fill(Z);
                }
                continue;
            }
            let c0 = h.col0[ii] as usize;
            let nb = h.disc_of(ii as i32).nb() as usize;
            apply_rule(self.column_rule(ii as i32, target), &mut D[c0..c0 + nb], S, c0);
            if target == Target::Field && kd == SurfaceKind::Interior {
                S[c0..c0 + nb].fill(Z);
                if let Some(m) = m.as_deref_mut() {
                    m[ii * NM..(ii + 1) * NM].fill(Z);
                }
            }
        }
    }
    fn value_at(cv: &[Complex64], q: &Pt) -> Complex64 {
        let mut v = Z;
        for t in 0..q.nk as usize {
            v += (q.wb[t] / q.w) * cv[q.col[t] as usize];
        }
        v
    }
    fn free_damper(&self, ii: i32) -> bool {
        let dm = self.h.pdmp[self.h.inst[ii as usize].patch as usize];
        dm >= 0 && self.h.dmp[dm as usize].free
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn incident(&mut self) -> Result<()> {
        let h = self.h;
        let (nh, NQ, NC, N, Ninst, K) = (self.nh, self.NQ, self.NC, self.N, self.Ninst, self.K);
        // incident wave at the inner points; scattered diffraction: the boundary values
        // projected on the basis (-dphi_I/dn on wet, L phi_I on pressure patches), by character
        let mut phiI = vec![Z; nh * NQ];
        let mut dphiI = vec![Z; phiI.len()];
        for hh in 0..nh {
            for qi in 0..NQ {
                let q = &h.inner[qi];
                let (phi, gr) = self.inc(hh, &q.x);
                phiI[hh * NQ + qi] = phi;
                // dI: dphi_I/dn on wet patches, -L phi_I on pressure patches (the Haskind integrand)
                let kd = h.kind_of(q.inst);
                // lids: -(K - kappa) phi_I, so that the Haskind integrand gains (K - kappa) phi_j phi_I
                dphiI[hh * NQ + qi] = if kd == SurfaceKind::Press {
                    -(gr[2] - K * phi)
                } else if kd == SurfaceKind::Lid {
                    -(K - self.fc.kappa[h.lid_of(q.inst) as usize]) * phi
                } else {
                    gr[0] * q.n[0] + gr[1] * q.n[1] + gr[2] * q.n[2]
                };
            }
        }
        self.phiI = phiI;
        self.dphiI = dphiI;
        if !self.pl.total && nh > 0 {
            let mut uS = vec![vec![vec![Z; N]; NC]; nh];
            let mut uSi = vec![vec![Vec::new(); Ninst]; nh];
            for hh in 0..nh {
                let mut vals = vec![Z; NQ];
                for qi in 0..NQ {
                    let kd = h.kind_of(h.inner[qi].inst);
                    if kd == SurfaceKind::Wet || kd == SurfaceKind::Plate {
                        vals[qi] = -self.dphiI[hh * NQ + qi]; // a plate: as a wetted patch (theory §9.7)
                    }
                    if kd == SurfaceKind::Press {
                        vals[qi] = -self.dphiI[hh * NQ + qi]; // L phi_S = -L phi_I: u = L phi_I
                    }
                }
                for ii in 0..Ninst {
                    project(h, ii as i32, &vals, &mut uSi[hh][ii])?;
                }
                for c in 0..NC {
                    for ii in 0..Ninst {
                        if h.inst[ii].image {
                            continue; // the same values as the original
                        }
                        let p = h.inst[ii].patch as usize;
                        let g = h.inst[ii].g as usize;
                        for l in 0..h.disc[p].nb() as usize {
                            uS[hh][c][h.dof0[ii] as usize + l] += h.chi[c][g] * uSi[hh][ii][l] / h.G as f64;
                        }
                    }
                }
            }
            self.uS = uS;
            self.uSi = uSi;
        }
        // the radiation columns: mode index, sign and whether the mode matches a character
        self.rmode = self.pl.rad.iter().map(|c| mode_index(h, c)).collect();
        Ok(())
    }

    // The rows of the outer point f (every patch kind but the sheets): its kernel row, the columns
    // of the other patch kinds, the dipole row of each character and the right-hand sides of its rows.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn row_data(&self, f: usize, rd: &mut RowData) {
        let h = self.h;
        let (NC, N, Ninst, NM, nrad, nh) = (self.NC, self.N, self.Ninst, self.NM, self.nrad, self.nh);
        let mut S = Vec::new();
        let mut D = Vec::new();
        let mut M = Vec::new();
        let o = &h.outer[f];
        rd.skip = self.sheet_like(o.inst); // collocation rows below
        if rd.skip {
            return;
        }
        match near4_mode(self.fc, h, &self.fin) {
            Some(false) => {
                kernel_row_near4::<false>(h, self.fc, &self.fin, &o.x, &self.so[f], self.need_s, &mut S, &mut D, &mut M)
            }
            Some(true) => {
                kernel_row_near4::<true>(h, self.fc, &self.fin, &o.x, &self.so[f], self.need_s, &mut S, &mut D, &mut M)
            }
            None => kernel_row(h, self.fc, &self.fin, &o.x, &self.so[f], self.need_s, &mut S, &mut D, &mut M),
        }
        // the column of each instance (the table at Target)
        self.columns(Target::Equation, &mut S, &mut D, Some(&mut M));
        let ko = h.kind_of(o.inst);
        rd.free_t = if ko == SurfaceKind::Interior {
            -4.0 * PI
        } else if h.z0(o.inst) || ko == SurfaceKind::Lid || ko == SurfaceKind::Plate {
            4.0 * PI
        } else {
            2.0 * PI
        };
        if h.wavemaker {
            // wavemakers in the walls: the dipoles of the coplanar patches vanish and those
            // of the coincident wall images (left out) cancel; their sources double those kept,
            // 4 pi phi = -2 int' G u, i.e. 2 pi phi = -int' G u with the kept instances
            D.fill(Z);
        }
        // the dipole row of each character: sum over the reflected instances
        rd.Dc.clear();
        rd.Dc.resize(NC * N, Z);
        for c in 0..NC {
            let dcv = &mut rd.Dc[c * N..(c + 1) * N];
            for ii in 0..Ninst {
                let p = h.inst[ii].patch as usize;
                let x = h.chi[c][h.inst[ii].g as usize];
                let c0 = h.col0[ii] as usize;
                let d0 = h.dof0[ii] as usize;
                for l in 0..h.disc[p].nb() as usize {
                    dcv[d0 + l] += x * D[c0 + l];
                }
            }
        }
        // the radiation sum of each mode over the instances, and the diffraction right-hand sides
        rd.vr.clear();
        rd.vr.resize(nrad, Z);
        for r in 0..nrad {
            let mut v = Z;
            for ii in 0..Ninst {
                v += M[ii * NM + self.rmode[r] as usize];
            }
            rd.vr[r] = v;
        }
        rd.vh.clear();
        rd.vh.resize(NC * nh, Z);
        for c in 0..NC {
            for hh in 0..nh {
                let mut v = Z;
                if self.pl.total {
                    for g in 0..h.G as usize {
                        let xg = v3(h.refl[g].sx * o.x[0], h.refl[g].sy * o.x[1], o.x[2]);
                        let (phi, _) = self.inc(hh, &xg);
                        v += h.chi[c][g] * phi;
                    }
                    v *= 4.0 * PI / h.G as f64;
                } else {
                    for ii in 0..Ninst {
                        let p = h.inst[ii].patch as usize;
                        let g = h.inst[ii].g as usize;
                        let c0 = h.col0[ii] as usize;
                        let d0 = h.dof0[ii] as usize;
                        for l in 0..h.disc[p].nb() as usize {
                            v -= h.chi[c][g] * S[c0 + l] * self.uS[hh][c][d0 + l];
                        }
                    }
                }
                rd.vh[c * nh + hh] = v;
            }
        }
    }

    // The rows r0 <= i < r1 of the system of character c gain the contributions of the outer points
    // f0 .. f0 + nb - 1, in the order of the points; match_[c * nrad + r]: the mode of column r is
    // even under character c. A: the rows r0.. of A_c (row-major, N columns), B: the same rows of
    // the right-hand sides B_c (row-major, nrhs columns), each item of body_rows its own rows.
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn add_rows(
        &self,
        f0: usize,
        nb: usize,
        rd: &[RowData],
        match_: &[bool],
        c: usize,
        r0: usize,
        A: &mut [Complex64],
        B: &mut [Complex64],
    ) {
        let h = self.h;
        let (N, nrad, nh, nrhs) = (self.N, self.nrad, self.nh, self.nrhs);
        let r1 = r0 + A.len() / N;
        for t in 0..nb {
            if rd[t].skip {
                continue;
            }
            let o = &h.outer[f0 + t];
            let dc = h.disc_of(o.inst);
            let mu = dc.mu();
            let row0 = h.dof0[o.inst as usize];
            let free_t = rd[t].free_t;
            for b in 0..dc.kv {
                for a in 0..dc.ku {
                    let i = (row0 + (o.iv + b) * mu + o.iu + a) as usize;
                    if i < r0 || i >= r1 {
                        continue;
                    }
                    let wi = o.w * o.bu[a as usize] * o.bv[b as usize];
                    // row i (row-major until the solve): only this item writes it
                    let Ar = &mut A[(i - r0) * N..(i - r0 + 1) * N];
                    for b2 in 0..dc.kv {
                        for a2 in 0..dc.ku {
                            let k = (row0 + (o.iv + b2) * mu + o.iu + a2) as usize;
                            Ar[k] += free_t * wi * o.bu[a2 as usize] * o.bv[b2 as usize];
                        }
                    }
                    let dcv = &rd[t].Dc[c * N..(c + 1) * N];
                    for k in 0..N {
                        Ar[k] -= wi * dcv[k];
                    }
                    let Br = &mut B[(i - r0) * nrhs..(i - r0 + 1) * nrhs];
                    for r in 0..nrad {
                        if !match_[c * nrad + r] {
                            continue;
                        }
                        Br[r] -= h.usign(self.rmode[r]) * wi * rd[t].vr[r];
                    }
                    for hh in 0..nh {
                        Br[nrad + hh] += wi * rd[t].vh[c * nh + hh];
                    }
                }
            }
        }
    }

    // The Galerkin rows of the outer points (all patches but the sheets): batches of points compute
    // their row data on the threads, then the threads add the batch to their own ranges of rows (of
    // each character), each row gaining the points in their order, so the system does not depend on
    // the thread count.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn body_rows(&mut self) -> Result<()> {
        let h = self.h;
        let (NC, N, nrad, nrhs, inner, work) = (self.NC, self.N, self.nrad, self.nrhs, self.inner, self.work);
        let mut match_ = vec![true; NC * nrad];
        for c in 0..NC {
            for r in 0..nrad {
                match_[c * nrad + r] = matches(h, c, self.rmode[r]);
            }
        }
        let nout = h.outer.len();
        let batch = (4 * inner) as usize;
        const ROWS: usize = 64;
        let nranges = N.div_ceil(ROWS);
        let mut rd: Vec<RowData> = (0..batch).map(|_| RowData::default()).collect();
        // the systems taken aside while the threads write them, the right-hand sides row-major
        // meanwhile (each range of rows its own elements; one character at a time transposed)
        let mut A = std::mem::take(&mut self.A);
        let mut Bt: Vec<Vec<Complex64>> = std::mem::take(&mut self.B)
            .into_iter()
            .map(|b| {
                let mut t = vec![Z; N * nrhs];
                for r in 0..nrhs {
                    for i in 0..N {
                        t[i * nrhs + r] = b[i + r * N];
                    }
                }
                t
            })
            .collect();
        {
            let this = &*self;
            let mut f0 = 0;
            while f0 < nout {
                let nb = batch.min(nout - f0);
                // each item its own row data
                rd[..nb].par_iter_mut().enumerate().try_for_each(|(t, r)| -> Result<()> {
                    work.check_cancel()?;
                    this.row_data(f0 + t, r);
                    Ok(())
                })?;
                // the ranges of ROWS rows of every character
                let mut ranges: Vec<(usize, usize, &mut [Complex64], &mut [Complex64])> = Vec::new();
                for (c, (a, bt)) in A.iter_mut().zip(Bt.iter_mut()).enumerate() {
                    let brows: Vec<&mut [Complex64]> = if nrhs > 0 {
                        bt.chunks_mut(ROWS * nrhs).collect()
                    } else {
                        (0..nranges).map(|_| <&mut [Complex64]>::default()).collect()
                    };
                    for (q, (ar, br)) in a.chunks_mut(ROWS * N).zip(brows).enumerate() {
                        ranges.push((c, q * ROWS, ar, br));
                    }
                }
                ranges.into_par_iter().try_for_each(|(c, r0, ar, br)| -> Result<()> {
                    work.check_cancel()?;
                    this.add_rows(f0, nb, &rd, &match_, c, r0, ar, br);
                    Ok(())
                })?;
                work.report(Stage::Assembly, (f0 + nb) as i32, nout as i32); // the progress
                f0 += batch;
            }
        }
        // the right-hand sides column-major again
        self.B = Bt
            .into_iter()
            .map(|t| {
                let mut b = vec![Z; N * nrhs.max(1)];
                for r in 0..nrhs {
                    for i in 0..N {
                        b[i + r * N] = t[i * nrhs + r];
                    }
                }
                b
            })
            .collect();
        self.A = A;
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn sheet_rows(&mut self) {
        // sheet rows: sum_n H x_n = 4 pi u + sum_wet K' u (total diffraction -4 pi dphi_I/dn)
        // in the Galerkin form of the body rows: per outer point of a sheet the test functions of its
        // panel (all coefficients free, also on the free edges); at omega = inf the free-surface
        // plates take these rows too (sheet_like)
        let h = self.h;
        let inf = self.fc.mode == FreqMode::Infinite;
        if !h.sheets && !(inf && self.se.iter().any(|e| e.plate)) {
            return;
        }
        let (NC, N, Ninst, NM, nrad, nh, nrhs) = (self.NC, self.N, self.Ninst, self.NM, self.nrad, self.nh, self.nrhs);
        let mut Hc = vec![Z; N];
        let se = self.se;
        // one evaluation point: the rows i of its test functions with their weights
        for ev in se {
            if ev.plate && !inf {
                continue; // the plate's own rows at the other frequencies (body_rows)
            }
            let sp = &ev.sp;
            let mut sr: SheetRowData = if inf { ev.inf.clone() } else { ev.reg.clone() };
            sheet_wave_row(h, self.fc, sp, &self.fin.in_, &self.fin.far, &mut sr);
            // the normal derivative of the column of each kind: the dipole column of the body rows
            // (the table at Target); the mode columns of the sheets are kept
            self.columns(Target::Equation, &mut sr.Kc, &mut sr.H, None);
            // the plates' mode columns at omega = inf: their single layer vanishes
            for ii in 0..Ninst {
                if inf && h.kind_of(ii as i32) == SurfaceKind::Plate {
                    sr.Mn[ii * NM..(ii + 1) * NM].fill(Z);
                }
            }
            for c in 0..NC {
                Hc.fill(Z);
                for ii in 0..Ninst {
                    let pp = h.inst[ii].patch as usize;
                    let xg = h.chi[c][h.inst[ii].g as usize];
                    for l in 0..h.disc[pp].nb() as usize {
                        Hc[h.dof0[ii] as usize + l] += xg * sr.H[h.col0[ii] as usize + l];
                    }
                }
                let mut rhs = vec![Z; nrhs];
                let dm = h.pdmp[h.inst[sp.inst as usize].patch as usize];
                let imperv = dm >= 0 && !h.dmp[dm as usize].porous; // psi_n = phi_n: no body velocity
                for r in 0..nrad {
                    let mode = self.rmode[r];
                    if !matches(h, c, mode) {
                        continue;
                    }
                    let mut v = if imperv { Z } else { Complex64::new(4.0 * PI * sp.nm[mode as usize], 0.0) };
                    for ii in 0..Ninst {
                        v += sr.Mn[ii * NM + mode as usize];
                    }
                    rhs[r] = h.usign(mode) * v;
                }
                for hh in 0..nh {
                    let mut dn = Z; // the character part of dphi_I/dn at x
                    for g in 0..h.G as usize {
                        let xg = v3(h.refl[g].sx * sp.x[0], h.refl[g].sy * sp.x[1], sp.x[2]);
                        let ng = v3(h.refl[g].sx * sp.n[0], h.refl[g].sy * sp.n[1], sp.n[2]);
                        let (_, gr) = self.inc(hh, &xg);
                        dn += h.chi[c][g] * (gr[0] * ng[0] + gr[1] * ng[1] + gr[2] * ng[2]);
                    }
                    dn /= h.G as f64;
                    let mut v = -4.0 * PI * dn;
                    if !self.pl.total {
                        for ii in 0..Ninst {
                            let pp = h.inst[ii].patch as usize;
                            let g = h.inst[ii].g as usize;
                            for l in 0..h.disc[pp].nb() as usize {
                                v += h.chi[c][g]
                                    * sr.Kc[h.col0[ii] as usize + l]
                                    * self.uS[hh][c][h.dof0[ii] as usize + l];
                            }
                        }
                    }
                    rhs[nrad + hh] = v;
                }
                for &(i, wi) in &ev.rw {
                    let i = i as usize;
                    let Ar = &mut self.A[c][i * N..(i + 1) * N];
                    for k in 0..N {
                        Ar[k] += wi * Hc[k];
                    }
                    for r in 0..nrhs {
                        self.B[c][i + r * N] += wi * rhs[r];
                    }
                }
            }
        }
        // dipole dampers: the unknown on a damper patch is psi_n with the jump
        // mu = c psi_n, c = -i eps / w; its columns are scaled by c in every row, and its rows gain
        // -4 pi psi_n in the Galerkin form (the mass matrix of the test functions)
        if !h.dmp.is_empty() && !self.limit {
            let mut cscale = vec![Complex64::new(1.0, 0.0); N];
            for &p in &h.rows {
                let dm = h.pdmp[h.inst[p as usize].patch as usize];
                if dm < 0 {
                    continue;
                }
                for l in 0..h.disc_of(p).nb() as usize {
                    cscale[h.dof0[p as usize] as usize + l] = Complex64::new(0.0, -h.dmp[dm as usize].eps / self.w);
                }
            }
            for c in 0..NC {
                for i in 0..N {
                    let Ar = &mut self.A[c][i * N..(i + 1) * N];
                    for k in 0..N {
                        Ar[k] *= cscale[k];
                    }
                }
            }
            for ev in se {
                if h.pdmp[h.inst[ev.sp.inst as usize].patch as usize] < 0 {
                    continue;
                }
                for c in 0..NC {
                    for &(i, wi) in &ev.rw {
                        for &(k, wk) in &ev.rw {
                            self.A[c][i as usize * N + k as usize] -= 4.0 * PI * wi * wk / ev.w;
                        }
                    }
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn unused_rows(&mut self) {
        // identity rows for the unused patches
        let h = self.h;
        let (NC, N, nrhs) = (self.NC, self.N, self.nrhs);
        for &p in &h.rows {
            if !self.unused(p) {
                continue;
            }
            for l in 0..h.disc_of(p).nb() as usize {
                let i = h.dof0[p as usize] as usize + l;
                for c in 0..NC {
                    for k in 0..N {
                        self.A[c][i * N + k] = Z;
                    }
                    self.A[c][i * N + i] = Complex64::new(1.0, 0.0);
                    for r in 0..nrhs {
                        self.B[c][i + r * N] = Z;
                    }
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn solve(&mut self) -> Result<()> {
        let h = self.h;
        let (NC, N, Ninst, nrhs, NQ) = (self.NC, self.N, self.Ninst, self.nrhs, self.NQ);
        // solve and expand: coefficients of every instance
        self.coef = vec![vec![Z; h.Ncol as usize]; nrhs];
        // the characters on the threads, at most inner at once (the row-major system freed once
        // copied), the blocks of their LUs on the same threads; the coefficients are then summed
        // over the characters in their order
        let mut sol: Vec<Vec<Complex64>> = vec![Vec::new(); NC];
        let work = self.work;
        if nrhs > 0 {
            let mut items: Vec<(&mut Vec<Complex64>, &mut Vec<Complex64>, &mut Vec<Complex64>)> =
                self.A.iter_mut().zip(self.B.iter_mut()).zip(sol.iter_mut()).map(|((a, b), s)| (a, b, s)).collect();
            try_for_each_grouped(&mut items, (self.inner as usize).min(NC), |(a, b, s)| {
                work.check_cancel()?;
                // each item its own character
                let mut Am = vec![Z; a.len()];
                **s = std::mem::take(&mut **b);
                for i in 0..N {
                    // row-major A to the column-major matrix
                    for k in 0..N {
                        Am[i + k * N] = a[i * N + k];
                    }
                }
                **a = Vec::new();
                if let Err(e) = lu_solve_c128(N as i32, &mut Am, nrhs as i32, &mut s[..], work) {
                    return Err(Error::new(e.kind(), format!("higher-order system: {}", e.message())));
                }
                Ok(())
            })?;
        }
        if nrhs > 0 {
            for c in 0..NC {
                let Bm = &sol[c];
                for r in 0..nrhs {
                    for ii in 0..Ninst {
                        let p = h.inst[ii].patch as usize;
                        let g = h.inst[ii].g as usize;
                        for l in 0..h.disc[p].nb() as usize {
                            self.coef[r][h.col0[ii] as usize + l] += h.chi[c][g] * Bm[h.dof0[ii] as usize + l + r * N];
                        }
                    }
                }
            }
        }
        if !h.dmp.is_empty() && !self.limit {
            // the solved psi_n becomes the jump mu = c psi_n
            for ii in 0..Ninst {
                let dm = h.pdmp[h.inst[ii].patch as usize];
                if dm < 0 {
                    continue;
                }
                let cj = Complex64::new(0.0, -h.dmp[dm as usize].eps / self.w);
                for r in 0..nrhs {
                    for l in 0..h.disc_of(ii as i32).nb() as usize {
                        self.coef[r][h.col0[ii] as usize + l] *= cj;
                    }
                }
            }
        }
        let mut phi_q = vec![vec![Z; NQ]; nrhs];
        for r in 0..nrhs {
            for qi in 0..NQ {
                phi_q[r][qi] = Self::value_at(&self.coef[r], &h.inner[qi]);
            }
        }
        self.phi_q = phi_q;
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn forces(&self, out: &mut Output) {
        // forces on every coordinate: A - iB/w = -rho s int n phi_j, X = i w rho s int n phi_D
        // with the generalized normals n and s = +1 (rigid), -1 (pressure modes)
        let h = self.h;
        let (D6, nrad, nh, NQ, rho, w) = (self.D6, self.nrad, self.nh, self.NQ, self.rho, self.w);
        for i in 0..D6 {
            let mi = mode_index(h, &self.pl.coords[i]);
            let si = h.usign(mi);
            let mi = mi as usize;
            for c in 0..nrad {
                let mut z = Z;
                for qi in 0..NQ {
                    if h.inst[h.inner[qi].inst as usize].image || self.free_damper(h.inner[qi].inst) {
                        continue;
                    }
                    let n = h.inner[qi].nm[mi];
                    if n != 0.0 {
                        z += n * self.phi_q[c][qi] * h.inner[qi].w;
                    }
                }
                z *= -rho * si;
                if self.limit {
                    let lim = if self.fc.mode == FreqMode::Zero { &mut out.A0 } else { &mut out.Ainf };
                    lim[i * nrad + c] = z.re as f32;
                    continue;
                }
                let o = i * nrad + c;
                out.A[o] = z.re as f32;
                out.B[o] = (-w * z.im) as f32;
            }
            for hh in 0..nh {
                let mut x = Z;
                let mut fk = Z;
                for qi in 0..NQ {
                    let n = h.inner[qi].nm[mi];
                    if n == 0.0 || h.inst[h.inner[qi].inst as usize].image || self.free_damper(h.inner[qi].inst) {
                        continue;
                    }
                    let pi = self.phiI[hh * NQ + qi];
                    let sheet = h.kind_of(h.inner[qi].inst) == SurfaceKind::Sheet; // the jump only
                    let pd = if self.pl.total || sheet {
                        self.phi_q[self.nrad + hh][qi]
                    } else {
                        self.phi_q[self.nrad + hh][qi] + pi
                    };
                    x += n * pd * h.inner[qi].w;
                    if !sheet {
                        fk += n * pi * h.inner[qi].w;
                    }
                }
                let o = hh * D6 + i;
                out.X[o] = to_cf(self.iwr * si * x);
                out.FK[o] = to_cf(self.iwr * si * fk);
                out.SC[o] = to_cf(self.iwr * si * (x - fk));
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn haskind(&self, out: &mut Output) {
        let h = self.h;
        let (nrad, nh, NQ) = (self.nrad, self.nh, self.NQ);
        if nh > 0 && nrad > 0 && !h.any_free_damper() {
            // no Haskind relation with a fixed damper (theory §9.5)
            for c in 0..nrad {
                let mi = self.rmode[c];
                let si = h.usign(mi);
                for hh in 0..nh {
                    let mut v = Z;
                    for qi in 0..NQ {
                        let q = &h.inner[qi];
                        if h.kind_of(q.inst) == SurfaceKind::Interior || h.inst[q.inst as usize].image {
                            continue;
                        }
                        // sheets: the n phi_I terms of the two faces cancel, the jump is left
                        let nt = if h.kind_of(q.inst) == SurfaceKind::Sheet { 0.0 } else { q.nm[mi as usize] };
                        v += (si * nt * self.phiI[hh * NQ + qi] - self.phi_q[c][qi] * self.dphiI[hh * NQ + qi]) * q.w;
                    }
                    out.HK[c * nh + hh] = to_cf(self.iwr * v);
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn surface_elevations(&self, out: &mut Output) {
        let h = self.h;
        let (nrad, nh, NQ, w, K) = (self.nrad, self.nh, self.NQ, self.w, self.K);
        // lid elevations eta = kappa phi / (i w) at the inner points of the lid patches (its Robin condition)
        if !self.limit && h.L > 0 {
            for qi in 0..NQ {
                let q = h.lid_out[qi];
                if q < 0 {
                    continue;
                }
                let q = q as usize;
                let kap = self.fc.kappa[h.lid_of(h.inner[qi].inst) as usize];
                for c in 0..nrad {
                    out.lrad[q * nrad + c] = to_cf(kap * self.phi_q[c][qi]);
                }
                if self.pl.total {
                    for hh in 0..nh {
                        out.ldif[q * nh + hh] = to_cf(kap * self.phi_q[nrad + hh][qi] * Complex64::new(0.0, -1.0 / w));
                    }
                }
            }
        }
        // pressure-surface elevations at the inner points of the pressure patches
        if !self.limit && h.Q > 0 {
            for qi in 0..NQ {
                let q = h.press_out[qi];
                if q < 0 {
                    continue;
                }
                let q = q as usize;
                for c in 0..nrad {
                    let nj = if self.rmode[c] >= h.P0 { h.inner[qi].nm[self.rmode[c] as usize] } else { 0.0 };
                    out.prad[q * nrad + c] = to_cf(K * self.phi_q[c][qi] + nj);
                }
                for hh in 0..nh {
                    let pd = if self.pl.total {
                        self.phi_q[nrad + hh][qi]
                    } else {
                        self.phi_q[nrad + hh][qi] + self.phiI[hh * NQ + qi]
                    };
                    out.pdif[q * nh + hh] = to_cf(K * pd * Complex64::new(0.0, -1.0 / w));
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn field_points(&mut self, out: &mut Output) {
        // field points (interior patches do not enter Green's identity in the fluid); the
        // potential of every point (with the velocity stencils) kept for the differences
        let h = self.h;
        let (nrad, nh) = (self.nrad, self.nh);
        self.Pall = if self.limit { 0 } else { h.fp.len() };
        self.P = if self.limit { 0 } else { h.Pu as usize };
        let Pall = self.Pall;
        let mut fpr = vec![Z; Pall * nrad];
        let mut fps = vec![Z; Pall * nh];
        // the control-surface points and stencils: also the fields at their reflections g x
        self.NCs = if self.pl.cdrift && !self.limit { Pall as i32 - h.cs0 } else { 0 };
        self.Gc = h.cs_G as usize;
        let Gc = self.Gc;
        let ncs = self.NCs.max(0) as usize;
        let mut fprg = vec![Z; ncs * Gc * nrad];
        let mut fpsg = vec![Z; ncs * Gc * nh];
        let mut cols = Columns::default();
        for q in 0..Pall {
            // the first half of the step (the field velocities the second)
            self.work.sub(q, 2 * Pall);
            let x = &h.fp[q];
            let (sf, need_s) = (&self.sf[q], self.need_s);
            match near4_mode(self.fc, h, &self.fin) {
                Some(false) => kernel_row_near4::<false>(
                    h,
                    self.fc,
                    &self.fin,
                    x,
                    sf,
                    need_s,
                    &mut cols.S,
                    &mut cols.D,
                    &mut cols.M,
                ),
                Some(true) => kernel_row_near4::<true>(
                    h,
                    self.fc,
                    &self.fin,
                    x,
                    sf,
                    need_s,
                    &mut cols.S,
                    &mut cols.D,
                    &mut cols.M,
                ),
                None => kernel_row(h, self.fc, &self.fin, x, sf, need_s, &mut cols.S, &mut cols.D, &mut cols.M),
            }
            self.columns(Target::Field, &mut cols.S, &mut cols.D, Some(&mut cols.M));
            if self.NCs > 0 && q as i32 >= h.cs0 {
                let o = (q - h.cs0 as usize) * Gc;
                self.reflected_fields(&cols, &mut fprg[o * nrad..(o + Gc) * nrad], &mut fpsg[o * nh..(o + Gc) * nh]);
            }
            self.radiation_field(q, &cols, &mut fpr[q * nrad..(q + 1) * nrad], out);
            self.scattered_field(q, &cols, &mut fps[q * nh..(q + 1) * nh], out);
        }
        self.fpr = fpr;
        self.fps = fps;
        self.fprg = fprg;
        self.fpsg = fpsg;
    }

    // The fields of a control-surface point at its reflections g x (g > 0) from its columns:
    // fprg[g * nrad + c], fpsg[g * nh + hh] with the solutions of the reflected instances.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn reflected_fields(&self, cols: &Columns, fprg: &mut [Complex64], fpsg: &mut [Complex64]) {
        let h = self.h;
        let (nrad, nh, Ninst, NM) = (self.nrad, self.nh, self.Ninst, self.NM);
        let Ncol = h.Ncol as usize;
        let (S, D, M) = (&cols.S, &cols.D, &cols.M);
        for gi in 1..self.Gc {
            let cg = &h.cs_col[gi];
            for c in 0..nrad {
                let mut v = Z;
                for k in 0..Ncol {
                    v += D[k] * self.coef[c][cg[k] as usize];
                }
                let mode = self.rmode[c];
                let mut mm = Z;
                for ii in 0..Ninst {
                    mm += M[ii * NM + mode as usize];
                }
                v -= h.usign(mode) * mode_sign(h, gi as i32, mode) * mm;
                fprg[gi * nrad + c] = v / (4.0 * PI);
            }
            for hh in 0..nh {
                let mut v = Z;
                for k in 0..Ncol {
                    v += D[k] * self.coef[nrad + hh][cg[k] as usize];
                }
                if !self.pl.total {
                    for ii in 0..Ninst {
                        let p = h.inst[ii].patch as usize;
                        let c0 = h.col0[ii] as usize;
                        let jj = h.cs_inst[gi][ii]; // the instance whose columns start at cg[c0]
                        for l in 0..h.disc[p].nb() as usize {
                            v -= S[c0 + l] * self.uSi[hh][jj][l];
                        }
                    }
                }
                fpsg[gi * nh + hh] = v / (4.0 * PI);
            }
        }
    }

    // The radiation potentials of field point q from its columns (fpr[c]); the points of the
    // problem also their elevations.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn radiation_field(&self, q: usize, cols: &Columns, fpr: &mut [Complex64], out: &mut Output) {
        let h = self.h;
        let (nrad, Ninst, NM, w) = (self.nrad, self.Ninst, self.NM, self.w);
        let Ncol = h.Ncol as usize;
        let (D, M) = (&cols.D, &cols.M);
        for c in 0..nrad {
            let mut v = Z;
            let mode = self.rmode[c];
            if h.wavemaker {
                // the dipoles cancel with the images left out, their sources double
                for ii in 0..Ninst {
                    v -= 2.0 * h.usign(mode) * M[ii * NM + mode as usize];
                }
            } else {
                for k in 0..Ncol {
                    v += D[k] * self.coef[c][k];
                }
                for ii in 0..Ninst {
                    v -= h.usign(mode) * M[ii * NM + mode as usize];
                }
            }
            fpr[c] = v / (4.0 * PI);
            if q < self.P {
                out.frad[q * nrad + c] = to_cf(w * w / self.pl.grav * v / (4.0 * PI));
            }
        }
    }

    // The scattered potentials of field point q from its columns (fps[hh]); the points of the
    // problem also the incident, scattered and total elevations.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn scattered_field(&self, q: usize, cols: &Columns, fps: &mut [Complex64], out: &mut Output) {
        let h = self.h;
        let (nrad, nh, Ninst, w) = (self.nrad, self.nh, self.Ninst, self.w);
        let Ncol = h.Ncol as usize;
        let (S, D) = (&cols.S, &cols.D);
        let x = &h.fp[q];
        for hh in 0..nh {
            let mut v = Z;
            for k in 0..Ncol {
                v += D[k] * self.coef[nrad + hh][k];
            }
            if !self.pl.total {
                for ii in 0..Ninst {
                    let p = h.inst[ii].patch as usize;
                    for l in 0..h.disc[p].nb() as usize {
                        v -= S[h.col0[ii] as usize + l] * self.uSi[hh][ii][l];
                    }
                }
            }
            let phiS = v / (4.0 * PI);
            fps[hh] = phiS;
            if q >= self.P {
                continue;
            }
            let (pI, _) = self.inc(hh, x);
            let etaI = pI * Complex64::new(0.0, -w / self.pl.grav);
            let etaS = phiS * Complex64::new(0.0, -w / self.pl.grav);
            let o = q * nh + hh;
            out.finc[o] = to_cf(etaI);
            out.fsca[o] = to_cf(etaS);
            out.ftot[o] = to_cf(etaI + etaS);
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn field_velocities(&self, out: &mut Output) {
        // the field velocities by differences of the potential, global axes
        let h = self.h;
        let (nrad, nh, P, w) = (self.nrad, self.nh, self.P, self.w);
        if h.fstep > 0.0 && P > 0 {
            let s = h.fstep;
            let cy = (h.yaw * PI / 180.0).cos();
            let sy = (h.yaw * PI / 180.0).sin();
            let glob = |v: &[Complex64; 3], o: &mut [Complex32]| {
                o[0] = to_cf(cy * v[0] - sy * v[1]);
                o[1] = to_cf(sy * v[0] + cy * v[1]);
                o[2] = to_cf(v[2]);
            };
            for q in 0..P {
                self.work.sub(P + q, 2 * P);
                let top = stencil_top(h.fp[q][2], s);
                let b = P + 6 * q;
                let grad = |f: &[Complex64], n: usize, c: usize, g: &mut [Complex64; 3]| {
                    let at = |k: usize| f[k * n + c];
                    g[0] = (at(b) - at(b + 1)) / (2.0 * s);
                    g[1] = (at(b + 2) - at(b + 3)) / (2.0 * s);
                    g[2] = if top {
                        (3.0 * at(q) - 4.0 * at(b + 4) + at(b + 5)) / (2.0 * s)
                    } else {
                        (at(b + 4) - at(b + 5)) / (2.0 * s)
                    };
                };
                for c in 0..nrad {
                    let mut g = [Z; 3];
                    grad(&self.fpr, nrad, c, &mut g);
                    for gd in &mut g {
                        *gd *= Complex64::new(0.0, w); // per unit displacement
                    }
                    let o = (q * nrad + c) * 3;
                    glob(&g, &mut out.fvr[o..o + 3]);
                }
                for hh in 0..nh {
                    let mut g = [Z; 3];
                    grad(&self.fps, nh, hh, &mut g);
                    let (_, gI) = self.inc(hh, &h.fp[q]);
                    let t = [gI[0] + g[0], gI[1] + g[1], gI[2] + g[2]];
                    let o = (q * nh + hh) * 3;
                    glob(&gI, &mut out.fvi[o..o + 3]);
                    glob(&g, &mut out.fvs[o..o + 3]);
                    glob(&t, &mut out.fvt[o..o + 3]);
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn kochin(&self, out: &mut Output) {
        // Kochin functions of the solutions on the global angles (theory §11.1):
        // Green's identity with the far-field kernel at the inner points, the patch kinds as for field points
        let h = self.h;
        if self.limit || self.pl.NK <= 0 {
            return;
        }
        let NK = self.pl.NK as usize;
        let Ncol = h.Ncol as usize;
        let nm = self.Ninst * self.NM;
        // the far-field columns and those of their derivative in the angle
        let mut cf = Columns { S: vec![Z; Ncol], D: vec![Z; Ncol], M: vec![Z; nm] };
        let mut ct = Columns { S: vec![Z; Ncol], D: vec![Z; Ncol], M: vec![Z; nm] };
        for t in 0..NK {
            let th = 2.0 * PI * t as f64 / NK as f64;
            let tf = th - h.yaw * PI / 180.0;
            self.far_columns(tf.cos(), tf.sin(), &mut cf, &mut ct);
            self.columns(Target::Field, &mut cf.S, &mut cf.D, Some(&mut cf.M));
            self.columns(Target::Field, &mut ct.S, &mut ct.D, Some(&mut ct.M));
            self.kochin_values(t, th, &cf, &ct, out);
        }
    }

    // The far-field columns of the direction (c, s) in the frame (cf) and of their derivative in
    // the angle (ct): Green's identity with the far-field kernel at the inner points.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn far_columns(&self, c: f64, s: f64, cf: &mut Columns, ct: &mut Columns) {
        let h = self.h;
        let NM = self.NM;
        cf.S.fill(Z);
        cf.D.fill(Z);
        ct.S.fill(Z);
        ct.D.fill(Z);
        cf.M.fill(Z);
        ct.M.fill(Z);
        for q in &h.inner {
            let f = far_kernel(self.fc, &q.x, &q.n, c, s);
            for k in 0..q.nk as usize {
                let col = q.col[k] as usize;
                cf.S[col] += f.g * q.wb[k];
                cf.D[col] += f.gn * q.wb[k];
                ct.S[col] += f.gt * q.wb[k];
                ct.D[col] += f.gnt * q.wb[k];
            }
            for k in 0..NM {
                if q.nm[k] != 0.0 {
                    cf.M[q.inst as usize * NM + k] += f.g * (q.w * q.nm[k]);
                    ct.M[q.inst as usize * NM + k] += f.gt * (q.w * q.nm[k]);
                }
            }
        }
    }

    // The Kochin functions of angle t (th) and their derivatives from the columns cf, ct.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn kochin_values(&self, t: usize, th: f64, cf: &Columns, ct: &Columns, out: &mut Output) {
        let h = self.h;
        let (nrad, nh, Ninst, NM) = (self.nrad, self.nh, self.Ninst, self.NM);
        let Ncol = h.Ncol as usize;
        for r in 0..nrad {
            let mut v = Z;
            let mut vt = Z;
            for k in 0..Ncol {
                v += cf.D[k] * self.coef[r][k];
                vt += ct.D[k] * self.coef[r][k];
            }
            let mode = self.rmode[r];
            for ii in 0..Ninst {
                v -= h.usign(mode) * cf.M[ii * NM + mode as usize];
                vt -= h.usign(mode) * ct.M[ii * NM + mode as usize];
            }
            (v, vt) = kochin_global(self.fc, h.px, h.py, th, v, vt);
            let o = t * nrad + r;
            out.kr[o] = to_cf(v);
            out.krt[o] = to_cf(vt);
        }
        for hh in 0..nh {
            let mut v = Z;
            let mut vt = Z;
            for k in 0..Ncol {
                v += cf.D[k] * self.coef[nrad + hh][k];
                vt += ct.D[k] * self.coef[nrad + hh][k];
            }
            if !self.pl.total {
                for ii in 0..Ninst {
                    for l in 0..h.disc_of(ii as i32).nb() as usize {
                        v -= cf.S[h.col0[ii] as usize + l] * self.uSi[hh][ii][l];
                        vt -= ct.S[h.col0[ii] as usize + l] * self.uSi[hh][ii][l];
                    }
                }
            }
            (v, vt) = kochin_global(self.fc, h.px, h.py, th, v, vt);
            let o = t * nh + hh;
            out.kd[o] = to_cf(v);
            out.kdt[o] = to_cf(vt);
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn body_outputs(&self, out: &mut Output) {
        // body pressure and velocity (theory §4.1) at the body output points: p / (rho g)
        // = (w^2/g) phi_j per unit displacement and -i w phi_D / g; the velocity from the
        // surface gradient of the spline and the normal velocity (n_j; scattered: the
        // incident gradient with its normal part removed), in global axes
        let h = self.h;
        let pl = self.pl;
        let (nrad, nh, w) = (self.nrad, self.nh, self.w);
        if (pl.bpres || pl.bvel) && !self.limit {
            let cy = (h.yaw * PI / 180.0).cos();
            let sy = (h.yaw * PI / 180.0).sin();
            let glob = |v: &[Complex64; 3], o: &mut [Complex32]| {
                o[0] = to_cf(cy * v[0] - sy * v[1]);
                o[1] = to_cf(sy * v[0] + cy * v[1]);
                o[2] = to_cf(v[2]);
            };
            let Nb = h.bpts.len();
            let iw = Complex64::new(0.0, w);
            for q in 0..Nb {
                let bp = &h.bpts[q];
                let in_ = &h.inst[bp.inst as usize];
                let press = h.kind_of(bp.inst) == SurfaceKind::Press;
                let mut xu = Vec3::ZERO;
                let mut xv = Vec3::ZERO;
                inst_tangents(h, in_, bp.u, bp.v, &mut xu, &mut xv);
                for c in 0..nrad {
                    let (mut f, mut fu, mut fv) = (Z, Z, Z);
                    let mut g = [Z; 3];
                    spline_values(h, bp.inst, &self.coef[c], bp.u, bp.v, bp.iu, bp.iv, &mut f, &mut fu, &mut fv);
                    if pl.bpres {
                        out.bpr[q * nrad + c] = to_cf(w * w / pl.grav * f);
                    }
                    if pl.bvel {
                        surface_gradient(&xu, &xv, fu, fv, &mut g);
                        // the normal velocity: n_j on wetted patches; on pressure patches (normal
                        // down) -(K phi + L phi) with L phi = the shape of a pressure mode
                        let rm = self.rmode[c];
                        let un = if press {
                            -(self.fc.K * f + (if rm >= h.P0 { bp.nm[rm as usize] } else { 0.0 }))
                        } else {
                            Complex64::new(h.usign(rm) * bp.nm[rm as usize], 0.0)
                        };
                        for k in 0..3 {
                            g[k] = iw * (g[k] + un * bp.n[k]);
                        }
                        let o = (q * nrad + c) * 3;
                        glob(&g, &mut out.bvr[o..o + 3]);
                    }
                }
                for hh in 0..nh {
                    let (mut f, mut fu, mut fv) = (Z, Z, Z);
                    let mut g = [Z; 3];
                    let mut pI = Z;
                    let mut gI = [Z; 3];
                    spline_values(
                        h,
                        bp.inst,
                        &self.coef[nrad + hh],
                        bp.u,
                        bp.v,
                        bp.iu,
                        bp.iv,
                        &mut f,
                        &mut fu,
                        &mut fv,
                    );
                    if !pl.total {
                        (pI, gI) = self.inc(hh, &bp.x);
                    }
                    if pl.bpres {
                        out.bpd[q * nh + hh] = to_cf(Complex64::new(0.0, -w / pl.grav) * (f + pI));
                    }
                    if pl.bvel {
                        surface_gradient(&xu, &xv, fu, fv, &mut g);
                        // the normal derivative of the solved potential: wetted -dphi_I/dn (scattered)
                        // or 0; pressure patches -(K phi - L phi_I) (L phi_S = -L phi_I), -K phi (total)
                        let gn = gI[0] * bp.n[0] + gI[1] * bp.n[1] + gI[2] * bp.n[2];
                        let dn = if press { -(self.fc.K * f - (gI[2] - self.fc.K * pI)) } else { -gn };
                        for k in 0..3 {
                            g[k] += gI[k] + dn * bp.n[k];
                        }
                        let o = (q * nh + hh) * 3;
                        glob(&g, &mut out.bvd[o..o + 3]);
                    }
                }
            }
        }
    }

    // The motions of every radiation column at heading hh (zero without the motions) and the
    // total potential of the heading (diffraction plus the radiation of the motions).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn motions(&self, out: &Output, hh: usize) -> (Vec<Complex64>, Vec<Complex64>) {
        let (nrad, D6) = (self.nrad, self.D6);
        let iw = Complex64::new(0.0, self.w);
        let mut Xr = vec![Z; nrad];
        if self.pl.rao {
            for r in 0..nrad {
                Xr[r] = to_cd(out.M[hh * D6 + self.pl.rad[r].coord as usize]);
            }
        }
        let mut cv = self.coef[nrad + hh].clone();
        for r in 0..nrad {
            if Xr[r] != Z {
                for k in 0..self.h.Ncol as usize {
                    cv[k] += iw * Xr[r] * self.coef[r][k];
                }
            }
        }
        (Xr, cv)
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn pressure_drift(&self, out: &mut Output) {
        // mean drift by pressure integration (theory §11.2): per body in its axes about
        // its origin, the pressure expanded about the mean wetted surface plus the waterline strip
        let h = self.h;
        let m = self.m;
        let pl = self.pl;
        let (nrad, nh, w, rho) = (self.nrad, self.nh, self.w, self.rho);
        if pl.pdrift && !self.limit && nh > 0 {
            let g = pl.grav;
            let iw = Complex64::new(0.0, w);
            // the first-order values of the quadratic QTFs, in the order of the drift
            let mut rec = if pl.pqtf.any() { Some(QtfRecorder::new(w, nh)) } else { None };
            for hh in 0..nh {
                if let Some(r) = rec.as_mut() {
                    r.heading(hh);
                }
                // the motions of every radiation column, the total potential of the heading
                let (Xr, cv) = self.motions(out, hh);
                // the total potential, its gradient and the normal velocity at a point (frame)
                let field = |ii: i32,
                             u: f64,
                             v: f64,
                             iu: i32,
                             iv: i32,
                             x: &Vec3,
                             n: &Vec3,
                             nm: &[f64],
                             phi: &mut Complex64,
                             grad: &mut [Complex64; 3]| {
                    let (mut f, mut fu, mut fv) = (Z, Z, Z);
                    spline_values(h, ii, &cv, u, v, iu, iv, &mut f, &mut fu, &mut fv);
                    let mut xu = Vec3::ZERO;
                    let mut xv = Vec3::ZERO;
                    inst_tangents(h, &h.inst[ii as usize], u, v, &mut xu, &mut xv);
                    surface_gradient(&xu, &xv, fu, fv, grad);
                    let mut un = Z; // normal velocity of the body point: i w sum xi_j n_j (about R)
                    for r in 0..nrad {
                        un += iw * Xr[r] * nm[self.rmode[r] as usize];
                    }
                    *phi = f;
                    if !pl.total {
                        // scattered: + phi_I, its gradient, minus its normal part (dphi_S/dn = -dphi_I/dn)
                        let (pI, gI) = self.inc(hh, x);
                        *phi += pI;
                        let gn = gI[0] * n[0] + gI[1] * n[1] + gI[2] * n[2];
                        for k in 0..3 {
                            grad[k] += gI[k] - gn * n[k];
                        }
                    }
                    for k in 0..3 {
                        grad[k] += un * n[k];
                    }
                };
                let rigid = |bi: usize| {
                    let mut X = [Z; 6];
                    for r in 0..nrad {
                        if pl.rad[r].body == bi as i32 && pl.rad[r].mode > 0 {
                            X[(pl.rad[r].mode - 1) as usize] = Xr[r];
                        }
                    }
                    X
                };
                // the motions of every body for the plate edges over the waterline (the band under
                // a plate rises with the plate's edge, of the wall's body or another), and a frame
                // point in the axes of a body
                let plated = h.wl.iter().any(|e| e.plate >= 0);
                let motions: Vec<PressureDrift> = if plated {
                    (0..h.NB as usize)
                        .map(|bi| PressureDrift::new(&rigid(bi), &m.bodies[bi].rotation_center, rho, g, w, false))
                        .collect()
                } else {
                    Vec::new()
                };
                let body_point = |bi: usize, x: &Vec3| {
                    let b = &m.bodies[bi];
                    let (yb, t) =
                        if h.single { (0.0, v3(0.0, 0.0, b.position[2])) } else { (b.yaw * PI / 180.0, b.position) };
                    let d = *x - t;
                    v3(yb.cos() * d[0] + yb.sin() * d[1], -yb.sin() * d[0] + yb.cos() * d[1], d[2])
                };
                for bi in 0..h.NB as usize {
                    let b = &m.bodies[bi];
                    let X = rigid(bi);
                    // frame vectors into the body axes (the single-body frame is the body's)
                    let yb = if h.single { 0.0 } else { b.yaw * PI / 180.0 };
                    let cb = yb.cos();
                    let sb = yb.sin();
                    let loc = |v: &Vec3| v3(cb * v[0] + sb * v[1], -sb * v[0] + cb * v[1], v[2]);
                    let locc = |v: &[Complex64; 3], o: &mut [Complex64; 3]| {
                        o[0] = cb * v[0] + sb * v[1];
                        o[1] = -sb * v[0] + cb * v[1];
                        o[2] = v[2];
                    };
                    // with pressure surfaces the hydrostatic part by the closed forms in the waterplane
                    // and the volume of the wetted and pressure patches, as in the low order
                    let closed = !b.pressure.is_empty();
                    let mut pd = PressureDrift::new(&X, &b.rotation_center, rho, g, w, closed);
                    if let Some(r) = rec.as_mut() {
                        r.group(bi, false, rho, closed, &b.rotation_center, b.position[2], &X);
                    }
                    // the mean wetted surface with the free-surface plates (P0 = 0 on them)
                    for q in &h.inner {
                        let in_ = &h.inst[q.inst as usize];
                        let kd = h.kind_of(q.inst);
                        if (kd != SurfaceKind::Wet && kd != SurfaceKind::Plate) || in_.image || in_.body != bi as i32 {
                            continue;
                        }
                        let mut phi = Z;
                        let mut grf = [Z; 3];
                        let mut gr = [Z; 3];
                        field(q.inst, q.uf, q.vf, q.iu, q.iv, &q.x, &q.n, &q.nm, &mut phi, &mut grf);
                        locc(&grf, &mut gr);
                        pd.surface(phi, &gr, &q.xl, &loc(&q.n), q.x[2], q.w); // q.xl: body coordinates about O
                        if let Some(r) = rec.as_mut() {
                            r.surface(phi, &gr, &q.xl, &loc(&q.n), q.x[2], q.w);
                        }
                    }
                    for e in &h.wl {
                        if h.inst[e.inst as usize].body != bi as i32 {
                            continue;
                        }
                        let mut phi = Z;
                        let mut gr = [Z; 3];
                        let mut nm = [0.0f64; MAX_MODES];
                        point_modes(h, e.inst, &e.xb, &e.nb, &mut nm);
                        field(e.inst, e.u, e.v, e.iu, e.iv, &e.x, &e.n, &nm, &mut phi, &mut gr);
                        if e.plate >= 0 {
                            // under a plate: the vertical displacement of the plate's edge there
                            let bp = e.plate as usize;
                            let top = if bp == bi {
                                motions[bp].vertical(&e.xb, None)
                            } else {
                                motions[bp].vertical(&body_point(bp, &e.x), None)
                            };
                            pd.waterline_under_plate(phi, top, &e.xb, &loc(&e.n), e.dl, None);
                            if let Some(r) = rec.as_mut() {
                                r.waterline_under_plate(phi, top, &e.xb, &loc(&e.n), e.dl, None);
                            }
                            continue;
                        }
                        pd.waterline(phi, &e.xb, &loc(&e.n), e.dl);
                        if let Some(r) = rec.as_mut() {
                            r.waterline(phi, &e.xb, &loc(&e.n), e.dl);
                        }
                    }
                    let mut crg = [0.0f64; 36];
                    if closed {
                        // the inner points of the wetted and pressure patches (body coordinates)
                        for q in &h.inner {
                            let in_ = &h.inst[q.inst as usize];
                            let kd = h.kind_of(q.inst);
                            if !in_.image
                                && in_.body == bi as i32
                                && (kd == SurfaceKind::Wet || kd == SurfaceKind::Press)
                            {
                                pd.hydrostatic_point(&q.xl, &loc(&q.n), q.x[2], q.w, kd == SurfaceKind::Wet);
                                if let Some(r) = rec.as_mut() {
                                    r.hydrostatic_point(&q.xl, &loc(&q.n), q.x[2], q.w, kd == SurfaceKind::Wet);
                                }
                            }
                        }
                        let hs = m.hydrostatics(bi as i32);
                        crg = hs.c_rho_g;
                    }
                    let o = (hh * h.NB as usize + bi) * 9;
                    let o9: &mut [f64; 9] = (&mut out.PD[o..o + 9]).try_into().unwrap();
                    pd.finish(b.position[2], &crg, o9);
                }
            }
            if let Some(r) = rec {
                out.qtf[0] = r.finish();
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn control_drift(&self, out: &mut Output) {
        // mean drift by the control surfaces: the total potential at the points and its
        // velocity by differences, the waterline and hydrostatic parts as in the pressure drift
        let h = self.h;
        let m = self.m;
        let pl = self.pl;
        let (nrad, nh, w, rho, D6, Gc) = (self.nrad, self.nh, self.w, self.rho, self.D6, self.Gc);
        if pl.cdrift && !self.limit && nh > 0 {
            let g = pl.grav;
            let s = h.csstep;
            let iw = Complex64::new(0.0, w);
            let Pc = h.cs.len();
            let Nr = h.cs_reps.len();
            let c0 = h.cs0 as usize;
            let st0 = c0 + Nr;
            // the field psi(y) = phi(g y) at fp index k (a representative or a stencil point)
            let val = |f: &[Complex64], fg: &[Complex64], n: usize, c: usize, k: usize, gi: usize| -> Complex64 {
                if gi == 0 { f[k * n + c] } else { fg[((k - c0) * Gc + gi) * n + c] }
            };
            // phi and grad phi at the point g x_rep: grad phi(g x) = R_g grad psi(x)
            let field_at = |f: &[Complex64],
                            fg: &[Complex64],
                            n: usize,
                            c: usize,
                            q: usize,
                            ph: &mut Complex64,
                            gr: &mut [Complex64; 3]| {
                let r = h.cs_rep[q] as usize;
                let gi = h.cs_g[q] as usize;
                let b = st0 + 6 * r;
                let x0 = c0 + r;
                let at = |k: usize| val(f, fg, n, c, k, gi);
                let top = stencil_top(h.cs[q].x[2], s);
                *ph = at(x0);
                gr[0] = h.refl[gi].sx * (at(b) - at(b + 1)) / (2.0 * s);
                gr[1] = h.refl[gi].sy * (at(b + 2) - at(b + 3)) / (2.0 * s);
                gr[2] = if top {
                    (3.0 * at(x0) - 4.0 * at(b + 4) + at(b + 5)) / (2.0 * s)
                } else {
                    (at(b + 4) - at(b + 5)) / (2.0 * s)
                };
            };
            // the control-surface values of the quadratic QTFs
            let mut rec = if pl.cqtf { Some(CsRecorder::new(nh)) } else { None };
            for hh in 0..nh {
                if let Some(r) = rec.as_mut() {
                    r.heading(hh);
                }
                let (Xr, cv) = self.motions(out, hh);
                for bi in 0..h.NB as usize {
                    let b = &m.bodies[bi];
                    let mut X = [Z; 6];
                    for r in 0..nrad {
                        if pl.rad[r].body == bi as i32 && pl.rad[r].mode > 0 {
                            X[(pl.rad[r].mode - 1) as usize] = Xr[r];
                        }
                    }
                    let yb = if h.single { 0.0 } else { b.yaw * PI / 180.0 };
                    let cb = yb.cos();
                    let sb = yb.sin();
                    let loc = |v: &Vec3| v3(cb * v[0] + sb * v[1], -sb * v[0] + cb * v[1], v[2]);
                    if let Some(r) = rec.as_mut() {
                        r.group(bi, rho, b.control_surface_alternative, &b.rotation_center, b.position[2], &X);
                    }
                    let mut cdr =
                        ControlSurfaceDrift::new(&X, &b.rotation_center, rho, g, w, b.control_surface_alternative);
                    for q in 0..Pc {
                        let cp = &h.cs[q];
                        if cp.body != bi as i32 {
                            continue;
                        }
                        let mut gs = [Z; 3];
                        let mut ps = Z;
                        let (pI, gI) = self.inc(hh, &cp.x);
                        field_at(&self.fps, &self.fpsg, nh, hh, q, &mut ps, &mut gs);
                        let mut ph = pI + ps;
                        let mut vf = [gI[0] + gs[0], gI[1] + gs[1], gI[2] + gs[2]];
                        for r in 0..nrad {
                            let xr = iw * Xr[r];
                            if xr == Z {
                                continue;
                            }
                            let mut gr = [Z; 3];
                            let mut pr = Z;
                            field_at(&self.fpr, &self.fprg, nrad, r, q, &mut pr, &mut gr);
                            ph += xr * pr;
                            for d in 0..3 {
                                vf[d] += xr * gr[d];
                            }
                        }
                        let vb = [cb * vf[0] + sb * vf[1], -sb * vf[0] + cb * vf[1], vf[2]];
                        if cp.part == 0 {
                            cdr.submerged(&vb, &cp.xb, &cp.nb, cp.w);
                        } else if cp.part == 1 {
                            cdr.free_surface(ph, &vb, &cp.xb, cp.w);
                        } else {
                            cdr.line(ph, &cp.xb, &cp.nb, cp.w);
                        }
                        if let Some(r) = rec.as_mut() {
                            r.point(cp.part, ph, &vb, &cp.xb, &cp.nb, cp.w);
                        }
                    }
                    // the body waterline bordered by free surface (under a free-surface plate, whose
                    // edge lies on the wall, S_f has no boundary: no line term)
                    for e in &h.wl {
                        if h.inst[e.inst as usize].body != bi as i32 || e.plate >= 0 {
                            continue;
                        }
                        let (mut f, mut fu, mut fv) = (Z, Z, Z);
                        spline_values(h, e.inst, &cv, e.u, e.v, e.iu, e.iv, &mut f, &mut fu, &mut fv);
                        if !pl.total {
                            let (pI, _) = self.inc(hh, &e.x);
                            f += pI;
                        }
                        cdr.waterline(f, &e.xb, &loc(&e.n), e.dl);
                        if let Some(r) = rec.as_mut() {
                            r.waterline(f, &e.xb, &loc(&e.n), e.dl);
                        }
                    }
                    // the hydrostatic part by the closed forms of theory §11.2 over the wetted patches
                    // and the plates (their area in the waterplane, no volume: z = 0)
                    let mut hs = PressureDrift::new(&X, &b.rotation_center, rho, g, w, true);
                    for q in &h.inner {
                        let in_ = &h.inst[q.inst as usize];
                        let kd = h.kind_of(q.inst);
                        if !in_.image && in_.body == bi as i32 && (kd == SurfaceKind::Wet || kd == SurfaceKind::Plate) {
                            hs.hydrostatic_point(&q.xl, &loc(&q.n), q.x[2], q.w, true);
                            if let Some(r) = rec.as_mut() {
                                r.hydrostatic_point(&q.xl, &loc(&q.n), q.x[2], q.w);
                            }
                        }
                    }
                    cdr.add(&hs.finish_about_o(b.position[2], &[0.0f64; 36]));
                    // the first-order force on the body for the moment about the moving rotation centre
                    let hy = m.hydrostatics(bi as i32);
                    let off = m.derived()[bi].coordinate_offset as usize;
                    let mut F1 = [Z; 3];
                    for d in 0..3 {
                        F1[d] = to_cd(out.X[hh * D6 + off + d]);
                        for r in 0..nrad {
                            let o = (off + d) * nrad + r;
                            F1[d] += (w * w * out.A[o] as f64 - iw * out.B[o] as f64) * Xr[r];
                        }
                        for k in 0..6 {
                            F1[d] -= hy.c_rho_g[d * 6 + k] * X[k];
                        }
                    }
                    let o = (hh * h.NB as usize + bi) * 9;
                    let o9: &mut [f64; 9] = (&mut out.CD[o..o + 9]).try_into().unwrap();
                    cdr.finish(&F1, o9);
                }
            }
            if let (Some(r), Some(s)) = (rec, out.qtf.first_mut()) {
                r.finish(s);
            }
        }
    }
}

/// One ordinary frequency or a limit (fi < 0): assembly, solve and outputs.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn solve_frequency(
    m: &Problem,
    h: &HoGeom,
    pl: &Plan,
    se: &[SheetEval],
    so: &[StaticRow],
    sf: &[StaticRow],
    fc: &FreqCtx,
    fi: i32,
    work: Work<'_>,
    out: &mut Output,
) -> Result<()> {
    let mut prof = Profile::new(); // phase timing on stderr with HB_PROFILE
    let mut f = Frequency::new(m, h, pl, se, so, sf, fc, work);
    f.incident()?;
    f.body_rows()?;
    f.sheet_rows();
    f.unused_rows();
    prof.mark("  ho assembly (rows)");
    f.solve()?;
    prof.mark("  ho LU solve");
    // the output steps, each reported and preceded by a cancel check
    let step = |k: i32| -> Result<()> {
        work.check_cancel()?;
        work.report(Stage::Outputs, k, 6);
        Ok(())
    };
    step(0)?;
    f.forces(out);
    f.haskind(out);
    f.surface_elevations(out);
    step(1)?;
    f.field_points(out);
    f.field_velocities(out);
    step(2)?;
    f.kochin(out);
    step(3)?;
    add_tank_terms(pl, fc.mode, fi, out); // internal tanks, before the motions
    // motions (theory §3.1; D x D with the pressure coordinates, no mass on them)
    solve_motions(m, pl, f.w, f.nh as i32, out, work)?;
    step(4)?;
    f.body_outputs(out);
    step(5)?;
    f.pressure_drift(out);
    f.control_drift(out);
    prof.mark("  ho outputs");
    Ok(())
}
