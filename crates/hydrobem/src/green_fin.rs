//! The finite-depth Green function (theory §8): Newman's decomposition with the tables of
//! scripts/gen_green_finite_tables.py, the eigenfunction series, per-frequency tiles.
//!
//! Finite-depth free-surface Green function, time factor
//! exp(i w t), depth h, K = w^2/g, H = K h, field point (z), source (zeta),
//! horizontal distance R, and the vertical distances
//!   v1 = |z - zeta|, v2 = z + zeta + 2h, v3 = |z + zeta|,
//!   v4 = z - zeta + 2h, v5 = -z + zeta + 2h, v6 = z + zeta + 4h,  r_p = sqrt(R^2 + v_p^2).
//!
//! R/h < 1/2 (Newman 1985; Wuillaume 2022 eqs. 12-20):
//!   Re G = sum_p 1/r_p + K [Ft(X, V3) + Ft(X, 2H - V1) + Ft(X, 2H + V1) + Ft(X, 2H + V2)]
//!          + (1/h) [Q(s, w1) + Q(s, w2)],   X = K R, V_p = K v_p, s = (R/h)^2, w_p = (v_p/h)^2,
//!   Q(s, w) = H I = sum_m s^m Q_m(w, H) from the tables of gen_green_finite_tables.py;
//! R/h >= 1/2 (John 1950, the eigenfunction series):
//!   Re G = -A0 E Y0(k0 R) + sum_n c_n cos k_n(z+h) cos k_n(zeta+h) K0(k_n R),
//! and in both regions Im G = -A0 E J0(k0 R), with
//!   A0 = 2 pi k0 / (1 - exp(-4 k0 h) + 4 k0 h exp(-2 k0 h)), E = sum_{p=3..6} exp(-k0 v_p),
//!   c_n = 4 (k_n^2 + K^2) / (k_n^2 h + K^2 h - K).
//! Limits: omega = 0 (gauge of Wuillaume eq. 67): sum_p 1/r_p + (1/h) [L0(s, w1) + L0(s, w2)],
//! series -(2/h)(log(R/2h) + gamma) + (4/h) sum cos cos K0 with k_n = n pi / h;
//! omega = inf: 1/r1 + 1/r2 - sum_{p=3..6} 1/r_p + (1/h) [Linf(s, w1) + Linf(s, w2)],
//! series (4/h) sum cos cos K0 with k_n = (n - 1/2) pi / h.
//!
//! Why this split, its table degrees, the per-frequency tiles and the four-pair kernels:
//! theory §8.2 and §8.3 (their grounds).

use std::sync::LazyLock;

use crate::constants::{GAMMA, LOG2, PI, fnv1a};
use crate::dispersion::{dispersion_x0, dispersion_xn};
use crate::error::{Error, Result};
use crate::green_inf::{
    AXIS_X, bessel_j01, bessel_j01_4, bessel_jy01, bessel_jy01_4, direct4_applies, near4_applies, wave_fx_over_x,
    wave_kernel_near4, wave_real, wave_real_direct4,
};
use crate::quadrature::{chebyshev_cosines, chebyshev_nodes, chebyshev_power_basis};
use crate::tables::green_fin_tables as T;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FreqMode {
    #[default]
    Ordinary,
    Zero,
    Infinite,
}

/// Regular part of the kernel and its derivatives:
///   Wr = Re G - sum_p sign_p / r_p,  Wi = Im G;
///   d/dR, d/dzeta (source) and d/dz (field point).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct Kern {
    pub Wr: f64,
    pub WrR: f64,
    pub WrZ: f64,
    pub Wrz: f64,
    pub Wi: f64,
    pub WiR: f64,
    pub WiZ: f64,
    pub Wiz: f64,
}

/// Second derivatives of the regular part: d2/dR2, (1/R) d/dR,
/// d2/dR dzeta, d2/dR dz and d2/dz dzeta, real and imaginary parts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct KernHess {
    pub RR: f64,
    pub R_R: f64,
    pub RZ: f64,
    pub Rz: f64,
    pub zZ: f64,
    pub iRR: f64,
    pub iR_R: f64,
    pub iRZ: f64,
    pub iRz: f64,
    pub izZ: f64,
}

/// Per-frequency data (dispersion roots, the tables summed over H).
#[derive(Clone, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct FiniteDepth {
    pub mode: FreqMode,
    pub h: f64,
    pub K: f64,
    pub H: f64,
    pub k0: f64,
    pub A0: f64,
    pub sign: [f64; 6], // signs of the Rankine terms sum_p sign_p / r_p
    pub kn: Vec<f64>,   // evanescent wave numbers and series coefficients
    pub cn: Vec<f64>,
    pub P: Vec<f64>, // [m][j] power-basis coefficients in y = w/2 - 1 of Q_m
    pub tiled: bool,
    pub T1: Vec<f64>,
    pub T2: Vec<f64>,
}

impl FiniteDepth {
    // per-frequency tiles of the smooth parts of the integral form:
    // TS x TW tiles of Phi1(xb, vb1), TS x TV of Phi2(xb, vb2), degree TD,
    // value and both derivatives each (theory §8.2, grounds: 4 x 4 tiles of degree 5 left
    // 6.6e-7 at H = 1000, 8 x 8 tiles or degree 7 reduce it 300-fold, and 8 x 8 of degree 5 is
    // the cheaper evaluation; the derivatives fitted separately because differentiating the value
    // fit amplified the table noise to 8e-6)
    pub const TS: i32 = 8;
    pub const TW: i32 = 8;
    pub const TV: i32 = 16;
    pub const TD: i32 = 5;
}

/// The number of evanescent roots k_n kept in the series form (and in `FinPoint::cs`).
pub const FIN_ROOTS: i32 = 24;

/// Per-frequency data of one point depth z (performance: reuse of kernel arguments): the
/// one-point factors of the propagating-mode exponentials, exp(-k0 v_p) = f_p(z) g_p(zeta),
/// and the depth factors of the evanescent series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinPoint {
    /// e^{k0 z}
    pub d: f64,
    /// e^{-k0 (z + h)}
    pub a: f64,
    /// e^{-k0 (h - z)}
    pub b: f64,
    /// e^{-k0 (z + 2h)}
    pub c: f64,
    /// cos, sin k_n (z + h) of the series (FIN_ROOTS roots)
    pub cs: [f64; 2 * FIN_ROOTS as usize],
}

impl Default for FinPoint {
    fn default() -> Self {
        FinPoint { d: 0.0, a: 0.0, b: 0.0, c: 0.0, cs: [0.0; 2 * FIN_ROOTS as usize] }
    }
}

/// The four exponentials exp(-k0 v_p), p = 3..6, of the propagating mode from the fin_point data
/// of the field point (pz) and of the source (pzeta): e3 = d d, e4 = a b, e5 = b a, e6 = c c.
#[inline]
pub fn propagating_products(pz: &FinPoint, pzeta: &FinPoint) -> [f64; 4] {
    [pz.d * pzeta.d, pz.a * pzeta.b, pz.b * pzeta.a, pz.c * pzeta.c]
}

/// The depth factor E = e3 + e4 + e5 + e6 of the propagating mode from its four exponentials, and
/// its derivatives: [E, dE/dzeta, dE/dz, d2E/dz dzeta].
#[inline]
pub fn propagating_factor(k0: f64, e: [f64; 4]) -> [f64; 4] {
    let [e3, e4, e5, e6] = e;
    [e3 + e4 + e5 + e6, k0 * (e3 + e4 - e5 - e6), k0 * (e3 - e4 + e5 - e6), k0 * k0 * (e3 - e4 - e5 + e6)]
}

// The sign of z - zeta (0 at z = zeta) and the six vertical distances v_p of the Rankine terms.
#[inline]
fn vertical_distances(z: f64, zeta: f64, h: f64) -> (f64, [f64; 6]) {
    let d1 = z - zeta;
    let sg = if d1 > 0.0 {
        1.0
    } else if d1 < 0.0 {
        -1.0
    } else {
        0.0
    };
    (sg, [d1.abs(), z + zeta + 2.0 * h, -(z + zeta), d1 + 2.0 * h, -d1 + 2.0 * h, z + zeta + 4.0 * h])
}

// The depth factor of a pair (propagating_factor; zero at the limits): from the fin_point data
// when both are given, else from the vertical distances.
#[inline]
fn pair_depth_factor(fd: &FiniteDepth, v: &[f64; 6], pts: Option<(&FinPoint, &FinPoint)>) -> [f64; 4] {
    if fd.mode != FreqMode::Ordinary {
        return [0.0; 4];
    }
    let e = match pts {
        Some((pz, pzeta)) => propagating_products(pz, pzeta),
        None => [(-fd.k0 * v[2]).exp(), (-fd.k0 * v[3]).exp(), (-fd.k0 * v[4]).exp(), (-fd.k0 * v[5]).exp()],
    };
    propagating_factor(fd.k0, e)
}

// Im G = -A0 E J0(k0 R) and its first derivatives, e = propagating_factor.
#[inline]
fn imaginary_part(fd: &FiniteDepth, e: &[f64; 4], j0: f64, j1: f64, o: &mut Kern) {
    o.Wi = -fd.A0 * e[0] * j0;
    o.WiR = fd.A0 * e[0] * fd.k0 * j1;
    o.WiZ = -fd.A0 * e[1] * j0;
    o.Wiz = -fd.A0 * e[2] * j0;
}

// The second derivatives of Im G (J1' = J0 - J1 / x, J1 / x -> 1/2 on the axis).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn imaginary_hess(fd: &FiniteDepth, e: &[f64; 4], R: f64, j0: f64, j1: f64, hs: &mut KernHess) {
    let k0 = fd.k0;
    let x = k0 * R;
    let j1x = if x > 1e-9 { j1 / x } else { 0.5 };
    hs.iRR = fd.A0 * e[0] * k0 * k0 * (j0 - j1x);
    hs.iR_R = fd.A0 * e[0] * k0 * k0 * j1x;
    hs.iRZ = fd.A0 * e[1] * k0 * j1;
    hs.iRz = fd.A0 * e[2] * k0 * j1;
    hs.izZ = -fd.A0 * e[3] * j0;
}

// The eigenfunction series sum_n c_n cos k_n(z + h) cos k_n(zeta + h) K0(k_n R) until
// k_n R >= SERIES_STOP and its derivatives [f, f_R, f_zeta, f_z] (k01, cs: as series_form); with
// HESS also its second derivatives into hs (K0'' = K0 + K1 / x).
#[inline(always)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn series_terms<const HESS: bool>(
    fd: &FiniteDepth,
    R: f64,
    k01: impl Fn(usize, f64) -> (f64, f64),
    cs: impl Fn(usize, f64) -> (f64, f64, f64, f64),
    hs: &mut KernHess,
) -> [f64; 4] {
    let mut s = [0.0f64; 4];
    for n in 0..fd.kn.len() {
        let k = fd.kn[n];
        let x = k * R;
        if x >= SERIES_STOP {
            break;
        }
        let (g0, g1) = k01(n, x);
        let ex = (-x).exp() / x.sqrt();
        let K0 = g0 * ex;
        let K1 = g1 * ex;
        let (cz, sz, cs, ss) = cs(n, k);
        let c = fd.cn[n];
        s[0] += c * cz * cs * K0;
        s[1] -= c * cz * cs * k * K1;
        s[2] -= c * cz * k * ss * K0;
        s[3] -= c * k * sz * cs * K0;
        if HESS {
            let k2 = k * k;
            hs.RR += c * cz * cs * k2 * (K0 + K1 / x);
            hs.R_R -= c * cz * cs * k2 * K1 / x;
            hs.RZ += c * k2 * cz * ss * K1;
            hs.Rz += c * k2 * sz * cs * K1;
            hs.zZ += c * k2 * sz * ss * K0;
        }
    }
    s
}

// The series form of the regular part at one pair (R >= h/2): the eigenfunction series
// sum_n c_n cos k_n(z + h) cos k_n(zeta + h) K0(k_n R) (until k_n R >= SERIES_STOP), at ordinary
// frequencies the propagating mode -A0 E (Y0, J0)(k0 R), at omega = 0 the term
// -(2/h)(log(R/2h) + gamma), minus the Rankine terms sign_p / r_p (integrated exactly as panel
// integrals). v, sg: vertical_distances; e: the depth factor (propagating_factor); jy: (J0, Y0,
// J1, Y1)(k0 R) at the ordinary frequencies; k01(n, x): the scaled (K0, K1) of root n at
// x = k_n R (scaled_k01); cs(n, k): cos, sin k (z + h) and cos, sin k (zeta + h). With HESS also
// the second derivatives into hs (K0'' = K0 + K1 / x, Y1' = Y0 - Y1 / x), in the same pass.
#[inline(always)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn series_form<const HESS: bool>(
    fd: &FiniteDepth,
    R: f64,
    v: &[f64; 6],
    sg: f64,
    e: &[f64; 4],
    jy: (f64, f64, f64, f64),
    k01: impl Fn(usize, f64) -> (f64, f64),
    cs: impl Fn(usize, f64) -> (f64, f64, f64, f64),
    hs: &mut KernHess,
) -> Kern {
    let h = fd.h;
    let [mut re, mut reR, mut reZ, mut rez] = series_terms::<HESS>(fd, R, k01, cs, hs);
    let mut o = Kern::default();
    if fd.mode == FreqMode::Ordinary {
        let [E, Ez, Ef, Ezf] = *e;
        let (j0, y0, j1, y1) = jy;
        re += -fd.A0 * E * y0;
        reR += fd.A0 * E * fd.k0 * y1;
        reZ += -fd.A0 * Ez * y0;
        rez += -fd.A0 * Ef * y0;
        imaginary_part(fd, e, j0, j1, &mut o);
        if HESS {
            let k0 = fd.k0;
            let x = k0 * R;
            hs.RR += fd.A0 * E * k0 * k0 * (y0 - y1 / x);
            hs.R_R += fd.A0 * E * k0 * k0 * y1 / x;
            hs.RZ += fd.A0 * Ez * k0 * y1;
            hs.Rz += fd.A0 * Ef * k0 * y1;
            hs.zZ -= fd.A0 * Ezf * y0;
            imaginary_hess(fd, e, R, j0, j1, hs);
        }
    } else if fd.mode == FreqMode::Zero {
        re += -2.0 / h * ((R / (2.0 * h)).ln() + GAMMA);
        reR += -2.0 / (h * R);
        if HESS {
            hs.RR += 2.0 / (h * R * R);
            hs.R_R -= 2.0 / (h * R * R);
        }
    }
    let dvz = [-sg, 1.0, -1.0, -1.0, 1.0, 1.0]; // d v_p / d zeta
    let dvf = [sg, 1.0, -1.0, 1.0, -1.0, 1.0]; // d v_p / d z
    for p in 0..6 {
        let r2 = R * R + v[p] * v[p];
        let r = r2.sqrt();
        let ir3 = fd.sign[p] / (r2 * r);
        re -= fd.sign[p] / r;
        reR += R * ir3;
        reZ += v[p] * dvz[p] * ir3;
        rez += v[p] * dvf[p] * ir3;
        if HESS {
            let ir5 = ir3 / r2;
            hs.RR += ir3 - 3.0 * R * R * ir5;
            hs.R_R += ir3;
            hs.RZ -= 3.0 * R * v[p] * dvz[p] * ir5;
            hs.Rz -= 3.0 * R * v[p] * dvf[p] * ir5;
            // v1 = |z - zeta|: d v1/d zeta d v1/d z = -sign^2 = -1 also at z = zeta (the limit)
            let vv = if p == 0 { -1.0 } else { dvz[p] * dvf[p] };
            hs.zZ += vv * (ir3 - 3.0 * v[p] * v[p] * ir5);
        }
    }
    o.Wr = re;
    o.WrR = reR;
    o.WrZ = reZ;
    o.Wrz = rez;
    o
}

// The (cz, sz, cs, ss) of root n (wave number k) of series_form: from the fin_point data when both
// are given, else from the depths z, zeta.
#[inline]
fn series_depths(
    pts: Option<(&FinPoint, &FinPoint)>,
    n: usize,
    k: f64,
    z: f64,
    zeta: f64,
    h: f64,
) -> (f64, f64, f64, f64) {
    match pts {
        Some((pz, pzeta)) => (pz.cs[2 * n], pz.cs[2 * n + 1], pzeta.cs[2 * n], pzeta.cs[2 * n + 1]),
        None => ((k * (z + h)).cos(), (k * (z + h)).sin(), (k * (zeta + h)).cos(), (k * (zeta + h)).sin()),
    }
}

const M: usize = T::M as usize;
const NC: usize = T::NW as usize + 1; // coefficients per power of s
// finite_series (the channel images) is used down to R = h / pi, the series form of finite_kernel
// from R = h / 2: at R >= h / pi, k_n R >= n - 1/2 exceeds SERIES_STOP from n = 19 on;
// FIN_ROOTS = 24 leaves a margin of 5 roots
const ROOTS: i32 = FIN_ROOTS;
// a term at k_n R >= 18 is K0(18) / K0(1/2) = 4.5e-9 / 0.92 of the first one at most (the
// R-derivative k_n / k_1 times more, about 1e-7): the tail stays far below the 2e-6 of theory §8
pub(crate) const SERIES_STOP: f64 = 18.0;
const NK: usize = T::NK as usize;

// Chebyshev coefficients in y -> power basis in y (degree T::NW), with the power coefficients a of
// the T_j (chebyshev_power_basis).
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn cheb_to_power(c: &[f64], a: &[[f64; NC]; NC], p: &mut [f64]) {
    for k in 0..NC {
        p[k] = c[0] * a[0][k] + c[1] * a[1][k];
    }
    for j in 2..NC {
        for k in 0..NC {
            p[k] += c[j] * a[j][k];
        }
    }
}

// Q_m coefficients (Chebyshev in y) of region r at the region variable x.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn region_coefficients(r: i32, x: f64, cw: &mut [f64]) {
    let meta = &T::REGION_META[5 * r as usize..5 * r as usize + 5];
    let a = meta[1];
    let b = meta[2];
    let n = meta[3] as usize;
    let c = &T::Q[meta[4] as usize..];
    let y = 2.0 * (x - a) / (b - a) - 1.0;
    let mut Tx = [0.0f64; 64];
    Tx[0] = 1.0;
    Tx[1] = y;
    for j in 2..=n {
        Tx[j] = 2.0 * y * Tx[j - 1] - Tx[j - 2];
    }
    for m in 0..=M {
        for jw in 0..NC {
            let mut s = 0.0;
            for jx in 0..=n {
                s += c[(m * (n + 1) + jx) * NC + jw] * Tx[jx];
            }
            cw[m * NC + jw] = s;
        }
    }
}

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn power_table(cw: &[f64], P: &mut Vec<f64>) {
    P.clear();
    P.resize((M + 1) * NC, 0.0);
    let a = chebyshev_power_basis::<NC>();
    for m in 0..=M {
        cheb_to_power(&cw[m * NC..(m + 1) * NC], &a, &mut P[m * NC..(m + 1) * NC]);
    }
}

// Q(s, w) = sum_m s^m Q_m(w), dQ/ds and dQ/dw at w = w1 and w2 together
// (independent Horner chains, vectorized by the compiler): (Q, dQ/ds, dQ/dw), each at w1 and w2.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn q_eval2(P: &[f64], s: f64, w1: f64, w2: f64) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let y = [0.5 * w1 - 1.0, 0.5 * w2 - 1.0];
    let mut v = [0.0f64; 2];
    let mut vs = [0.0f64; 2];
    let mut vy = [0.0f64; 2];
    for m in (0..=M).rev() {
        let c = &P[m * NC..(m + 1) * NC];
        let mut p = [c[NC - 1], c[NC - 1]];
        let mut dp = [0.0f64; 2];
        for k in (0..NC - 1).rev() {
            for i in 0..2 {
                dp[i] = dp[i] * y[i] + p[i];
                p[i] = p[i] * y[i] + c[k];
            }
        }
        for i in 0..2 {
            vs[i] = vs[i] * s + v[i];
            v[i] = v[i] * s + p[i];
            vy[i] = vy[i] * s + dp[i];
        }
    }
    let mut qw = [0.0f64; 2];
    for i in 0..2 {
        qw[i] = 0.5 * vy[i];
    }
    (v, vs, qw)
}

// (e^x sqrt(x) K0(x), e^x sqrt(x) K1(x)) for T::K_LO <= x < 48 (the
// tiles [1/2, 1), [1, 2), ..., [47, 48) of T::BESSEL_K; the series stop at SERIES_STOP).
#[inline]
fn scaled_k01(x: f64) -> (f64, f64) {
    let i: usize;
    let t: f64;
    if x < 1.0 {
        i = 0;
        t = 2.0 * (x - T::K_LO) / (1.0 - T::K_LO) - 1.0;
    } else {
        let ii = x as i32;
        i = ii as usize;
        t = 2.0 * (x - ii as f64) - 1.0;
    }
    let off = i * (NK + 1) * 2;
    let c = &T::BESSEL_K[off..off + (NK + 1) * 2];
    let mut a0 = 0.0;
    let mut a1 = 0.0;
    for j in (0..=NK).rev() {
        a0 = a0 * t + c[2 * j];
        a1 = a1 * t + c[2 * j + 1];
    }
    (a0, a1)
}

// Per-frequency tiles of the two smooth parts of the integral form, dimensionless (times 1/h in the kernel):
//   Phi1(xb, vb1) = H [Ft(X, 2H - V1) + Ft(X, 2H + V1)] + Q(xb^2, vb1^2),  vb1 = v1/h in [0, 1],
//   Phi2(xb, vb2) = H Ft(X, 2H + V2) + Q(xb^2, vb2^2),                    vb2 = v2/h in [0, 2],
// xb = R/h in [0, 1/2], X = H xb, V_p = H vb_p (the Ft terms only at ordinary
// frequencies). Phi1 is even in vb1 and both are even in xb. The value and
// its two derivatives are interpolated separately from their direct values
// (Chebyshev, degree TD, first-kind nodes per tile, power basis), so that
// the derivatives keep the accuracy of the direct derivatives.
const XMAX: f64 = 0.5;
const TN: usize = FiniteDepth::TD as usize + 1;

// Phi, dPhi/dxb, dPhi/dvb directly.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn phi_direct(fd: &FiniteDepth, which: i32, xb: f64, vb: f64) -> [f64; 3] {
    let (q, qs, qw) = q_eval2(&fd.P, xb * xb, vb * vb, vb * vb);
    let mut out = [q[0], 2.0 * xb * qs[0], 2.0 * vb * qw[0]];
    if fd.mode != FreqMode::Ordinary {
        return out;
    }
    let H = fd.H;
    let X = H * xb;
    let H2 = H * H;
    if which == 1 {
        let (F, FX, FY) = wave_real(X, H * (2.0 - vb));
        out[0] += H * F;
        out[1] += H2 * FX;
        out[2] -= H2 * FY;
        let (F, FX, FY) = wave_real(X, H * (2.0 + vb));
        out[0] += H * F;
        out[1] += H2 * FX;
        out[2] += H2 * FY;
    } else {
        let (F, FX, FY) = wave_real(X, H * (2.0 + vb));
        out[0] += H * F;
        out[1] += H2 * FX;
        out[2] += H2 * FY;
    }
    out
}

#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn build_tiles(fd: &mut FiniteDepth) {
    const N: usize = TN;
    let t = chebyshev_nodes::<N>();
    // Chebyshev -> power basis: T_j(t) = sum_k a[j][k] t^k
    let a = chebyshev_power_basis::<N>();
    let cs = chebyshev_cosines::<N>(); // cos(pi j (i + 1/2) / n)
    let ts = FiniteDepth::TS as usize;
    for which in 1..=2 {
        let ny = if which == 1 { FiniteDepth::TW } else { FiniteDepth::TV } as usize;
        let ymax = if which == 1 { 1.0 } else { 2.0 };
        let mut out = vec![0.0f64; ts * ny * 3 * N * N];
        let dx = XMAX / FiniteDepth::TS as f64;
        let dy = ymax / ny as f64;
        for ix in 0..ts {
            for iy in 0..ny {
                let mut f = [[[0.0f64; N]; N]; 3];
                for i in 0..N {
                    for j in 0..N {
                        let v = phi_direct(
                            fd,
                            which,
                            dx * (ix as f64 + 0.5 * (t[i] + 1.0)),
                            dy * (iy as f64 + 0.5 * (t[j] + 1.0)),
                        );
                        for q in 0..3 {
                            f[q][i][j] = v[q];
                        }
                    }
                }
                for q in 0..3 {
                    let mut c = [[0.0f64; N]; N];
                    for ji in 0..N {
                        for jj in 0..N {
                            let mut acc = 0.0;
                            for i in 0..N {
                                for j in 0..N {
                                    acc += f[q][i][j] * cs[ji][i] * cs[jj][j];
                                }
                            }
                            c[ji][jj] = acc * (if ji != 0 { 2.0 } else { 1.0 }) * (if jj != 0 { 2.0 } else { 1.0 })
                                / (N * N) as f64;
                        }
                    }
                    let off = ((ix * ny + iy) * 3 + q) * N * N;
                    let pw = &mut out[off..off + N * N];
                    for ji in 0..N {
                        for jj in 0..N {
                            for ki in 0..=ji {
                                for kj in 0..=jj {
                                    pw[ki * N + kj] += c[ji][jj] * a[ji][ki] * a[jj][kj];
                                }
                            }
                        }
                    }
                }
            }
        }
        if which == 1 {
            fd.T1 = out;
        } else {
            fd.T2 = out;
        }
    }
    fd.tiled = true;
}

// The tile of (xb, vb): its coefficients and the tile variables.
#[inline]
fn tile_of(fd: &FiniteDepth, which: i32, xb: f64, vb: f64) -> (&[f64], f64, f64, usize) {
    const N: usize = TN;
    let ny = if which == 1 { FiniteDepth::TW } else { FiniteDepth::TV };
    let ymax = if which == 1 { 1.0 } else { 2.0 };
    let ux = xb * (FiniteDepth::TS as f64 / XMAX);
    let uy = vb * (ny as f64 / ymax);
    let ix = (ux as i32).min(FiniteDepth::TS - 1);
    let iy = (uy as i32).min(ny - 1);
    let tx = 2.0 * (ux - ix as f64) - 1.0;
    let ty = 2.0 * (uy - iy as f64) - 1.0;
    let off = (ix as usize * ny as usize + iy as usize) * 3 * N * N;
    let c = &(if which == 1 { &fd.T1 } else { &fd.T2 })[off..off + 3 * N * N];
    (c, tx, ty, ny as usize)
}

// Phi, dPhi/dxb, dPhi/dvb from the tiles.
#[inline]
fn phi_tiles(fd: &FiniteDepth, which: i32, xb: f64, vb: f64) -> [f64; 3] {
    const N: usize = TN;
    let (c, tx, ty, _) = tile_of(fd, which, xb, vb);
    let mut v = [0.0f64; 3];
    for i in (0..N).rev() {
        for q in 0..3 {
            let r = &c[q * N * N + i * N..q * N * N + i * N + N];
            let mut p = r[N - 1];
            for j in (0..N - 1).rev() {
                p = p * ty + r[j];
            }
            v[q] = v[q] * tx + p;
        }
    }
    v
}

// phi_tiles at four points: when the four lie in one tile, its Horner chains in the four lanes
// (AVX2) on the coefficients of that tile, every operation the scalar one in the same order (no
// fused multiply-add), so equal to phi_tiles to the bit; otherwise phi_tiles point by point.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn phi_tiles4(fd: &FiniteDepth, which: i32, xb: &[f64; 4], vb: &[f64; 4]) -> [[f64; 3]; 4] {
    use std::arch::x86_64::*;
    const N: usize = TN;
    let t: [(&[f64], f64, f64, usize); 4] = std::array::from_fn(|k| tile_of(fd, which, xb[k], vb[k]));
    if !(1..4).all(|k| std::ptr::eq(t[k].0.as_ptr(), t[0].0.as_ptr())) {
        return std::array::from_fn(|k| phi_tiles(fd, which, xb[k], vb[k]));
    }
    let c = t[0].0;
    let (tx, ty): ([f64; 4], [f64; 4]) = (std::array::from_fn(|k| t[k].1), std::array::from_fn(|k| t[k].2));
    let mut o = [[0.0f64; 4]; 3];
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); the loads and stores read
    // and write arrays of four doubles.
    unsafe {
        let b = |v: f64| _mm256_set1_pd(v);
        let (tx, ty) = (_mm256_loadu_pd(tx.as_ptr()), _mm256_loadu_pd(ty.as_ptr()));
        let mut v = [_mm256_setzero_pd(); 3];
        for i in (0..N).rev() {
            for q in 0..3 {
                let r = &c[q * N * N + i * N..q * N * N + i * N + N];
                let mut p = b(r[N - 1]);
                for j in (0..N - 1).rev() {
                    p = _mm256_add_pd(_mm256_mul_pd(p, ty), b(r[j]));
                }
                v[q] = _mm256_add_pd(_mm256_mul_pd(v[q], tx), p);
            }
        }
        for q in 0..3 {
            _mm256_storeu_pd(o[q].as_mut_ptr(), v[q]);
        }
    }
    std::array::from_fn(|k| [o[0][k], o[1][k], o[2][k]])
}

/// Whether finite_kernel4 holds four pairs at the horizontal distances R: all four in the integral
/// form from the per-frequency tiles (R < h/2, a tiled kernel), or all four in the series form
/// (R >= h/2).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn fin4_applies(fd: &FiniteDepth, R: &[f64; 4]) -> bool {
    (fd.tiled && R.iter().all(|&r| r < 0.5 * fd.h)) || R.iter().all(|&r| r >= 0.5 * fd.h)
}

// scaled_k01 at the four x (lanes with live false: any value, not used), with AVX2: the tile and
// its variable per lane as scaled_k01 finds them, the Horner chains in the four lanes on the
// coefficients of each lane's tile (gathered), every operation the scalar one in the same order.
#[inline]
fn scaled_k01_4(x: &[f64; 4], live: &[bool; 4]) -> ([f64; 4], [f64; 4]) {
    use std::arch::x86_64::*;
    let mut off = [0i64; 4];
    let mut tv = [0.0f64; 4];
    for k in 0..4 {
        if !live[k] {
            continue; // tile 0, t = 0: inside the table
        }
        let (i, t): (usize, f64);
        if x[k] < 1.0 {
            i = 0;
            t = 2.0 * (x[k] - T::K_LO) / (1.0 - T::K_LO) - 1.0;
        } else {
            let ii = x[k] as i32;
            i = ii as usize;
            t = 2.0 * (x[k] - ii as f64) - 1.0;
        }
        off[k] = (i * (NK + 1) * 2) as i64;
        tv[k] = t;
    }
    assert!(off.iter().all(|&o| o as usize + (NK + 1) * 2 <= T::BESSEL_K.len()));
    let (mut g0, mut g1) = ([0.0f64; 4], [0.0f64; 4]);
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); every gathered index is
    // within T::BESSEL_K (the assertion above); the loads and stores read and write arrays of four.
    unsafe {
        let base = T::BESSEL_K.as_ptr();
        let o = _mm256_loadu_si256(off.as_ptr() as *const __m256i);
        let t = _mm256_loadu_pd(tv.as_ptr());
        let (mut a0, mut a1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        for j in (0..=NK).rev() {
            let i0 = _mm256_add_epi64(o, _mm256_set1_epi64x(2 * j as i64));
            let i1 = _mm256_add_epi64(o, _mm256_set1_epi64x(2 * j as i64 + 1));
            a0 = _mm256_add_pd(_mm256_mul_pd(a0, t), _mm256_i64gather_pd::<8>(base, i0));
            a1 = _mm256_add_pd(_mm256_mul_pd(a1, t), _mm256_i64gather_pd::<8>(base, i1));
        }
        _mm256_storeu_pd(g0.as_mut_ptr(), a0);
        _mm256_storeu_pd(g1.as_mut_ptr(), a1);
    }
    (g0, g1)
}

// finite_kernel's series form (R >= h/2) at four pairs with the fin_point data: the scaled K0, K1
// of every root four pairs at a time (scaled_k01_4), (J0, Y0, J1, Y1)(k0 R) by bessel_jy01_4 where
// the four share a tile; the sums per pair by series_form, as finite_kernel's.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn finite_series4(
    fd: &FiniteDepth,
    R: &[f64; 4],
    z: f64,
    zeta: &[f64; 4],
    pz: &FinPoint,
    pzeta: [&FinPoint; 4],
) -> [Kern; 4] {
    let h = fd.h;
    let nr = fd.kn.len().min(ROOTS as usize);
    // the roots of each pair before its stop (series_form's break)
    let stop: [usize; 4] = std::array::from_fn(|k| (0..nr).find(|&n| fd.kn[n] * R[k] >= SERIES_STOP).unwrap_or(nr));
    let nm = *stop.iter().max().unwrap();
    let mut g = [([0.0f64; 4], [0.0f64; 4]); ROOTS as usize];
    for n in 0..nm {
        let x: [f64; 4] = std::array::from_fn(|k| fd.kn[n] * R[k]);
        g[n] = scaled_k01_4(&x, &std::array::from_fn(|k| n < stop[k]));
    }
    let ordinary = fd.mode == FreqMode::Ordinary;
    let kr: [f64; 4] = std::array::from_fn(|k| fd.k0 * R[k]);
    let jy4 = if ordinary { bessel_jy01_4(&kr) } else { None };
    std::array::from_fn(|k| {
        let (sg, v) = vertical_distances(z, zeta[k], h);
        let pts = Some((pz, pzeta[k]));
        let e = pair_depth_factor(fd, &v, pts);
        let jy = match (&jy4, ordinary) {
            (Some(a), _) => a[k],
            (None, true) => bessel_jy01(kr[k]),
            (None, false) => (0.0, 0.0, 0.0, 0.0),
        };
        series_form::<false>(
            fd,
            R[k],
            &v,
            sg,
            &e,
            jy,
            |n, _| (g[n].0[k], g[n].1[k]),
            |n, kk| series_depths(pts, n, kk, z, zeta[k], h),
            &mut KernHess::default(),
        )
    })
}

/// finite_kernel at four pairs of one field point (depth z, fin_point data pz) and four sources
/// (depths zeta, fin_point data pzeta) at the horizontal distances R, all four in the tiled
/// integral form or all four in the series form (fin4_applies), with AVX2. Integral form:
/// phi_tiles (where the four share a tile), Ft(X, V3) (wave_kernel_near4 or wave_real_direct4,
/// where the four share a tile) and J0, J1 (bessel_j01_4) four points at a time; series form: the
/// scaled K0, K1 of every root (gathered tiles) and J0, Y0, J1, Y1 (bessel_jy01_4); the rest point
/// by point. Every operation is finite_kernel's in the same order, so the result equals
/// finite_kernel's (with the fin_point data) to the bit.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn finite_kernel4(
    fd: &FiniteDepth,
    R: &[f64; 4],
    z: f64,
    zeta: &[f64; 4],
    pz: &FinPoint,
    pzeta: [&FinPoint; 4],
) -> [Kern; 4] {
    debug_assert!(fin4_applies(fd, R));
    let h = fd.h;
    if R[0] >= 0.5 * h {
        return finite_series4(fd, R, z, zeta, pz, pzeta);
    }
    let ordinary = fd.mode == FreqMode::Ordinary;
    let dv: [(f64, [f64; 6]); 4] = std::array::from_fn(|k| vertical_distances(z, zeta[k], h));
    let ih = 1.0 / h;
    let ih2 = ih * ih;
    let xb: [f64; 4] = std::array::from_fn(|k| R[k] * ih);
    let f1 = phi_tiles4(fd, 1, &xb, &std::array::from_fn(|k| dv[k].1[0] * ih));
    let f2 = phi_tiles4(fd, 2, &xb, &std::array::from_fn(|k| dv[k].1[1] * ih));
    let mut o = [Kern::default(); 4];
    let mut Bz = [0.0f64; 4];
    let mut A1z = [0.0f64; 4];
    for k in 0..4 {
        o[k].Wr = ih * (f1[k][0] + f2[k][0]);
        o[k].WrR = ih2 * (f1[k][1] + f2[k][1]);
        A1z[k] = -dv[k].0 * ih2 * f1[k][2]; // d vb1 / d zeta = -sign(z - zeta) / h
        Bz[k] = ih2 * f2[k][2];
    }
    if ordinary {
        let K = fd.K;
        // K Ft(X, Y) at X = K R, Y = K v3 (wave_real; its near branch four at a time)
        let X: [f64; 4] = std::array::from_fn(|k| K * R[k]);
        let Y: [f64; 4] = std::array::from_fn(|k| K * dv[k].1[2]);
        let coincident = (0..4).any(|k| R[k] == 0.0 && dv[k].1[2] == 0.0);
        let F3: [(f64, f64, f64); 4] = if !coincident && near4_applies(&X, &Y) {
            let w = wave_kernel_near4(&X, &Y);
            std::array::from_fn(|k| (w[k].F, w[k].FX, w[k].FY))
        } else if !coincident && direct4_applies(&X, &Y) {
            wave_real_direct4(&X, &Y)
        } else {
            std::array::from_fn(|k| wave_real_or_limit(K, R[k], dv[k].1[2]))
        };
        let kr: [f64; 4] = std::array::from_fn(|k| fd.k0 * R[k]);
        let (j0, j1) = bessel_j01_4(&kr).unwrap_or_else(|| {
            let j: [(f64, f64); 4] = std::array::from_fn(|k| bessel_j01(kr[k]));
            (std::array::from_fn(|k| j[k].0), std::array::from_fn(|k| j[k].1))
        });
        for k in 0..4 {
            let (F3, F3X, F3Y) = F3[k];
            o[k].Wr += K * F3;
            o[k].WrR += K * K * F3X;
            Bz[k] -= K * K * F3Y;
            let e = propagating_factor(fd.k0, propagating_products(pz, pzeta[k]));
            imaginary_part(fd, &e, j0[k], j1[k], &mut o[k]);
        }
    }
    for k in 0..4 {
        o[k].WrZ = A1z[k] + Bz[k];
        o[k].Wrz = -A1z[k] + Bz[k];
    }
    o
}

// Ft(X, Y) and its derivatives at X = K R, Y = K v3 (wave_real), and at coincident points on
// z = 0 (R = v3 = 0) the limit of Ft + 2 psi, psi = log((Y + R0)/K), with no derivatives.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn wave_real_or_limit(K: f64, R: f64, v3: f64) -> (f64, f64, f64) {
    if R == 0.0 && v3 == 0.0 { (-2.0 * (GAMMA - LOG2 + K.ln()), 0.0, 0.0) } else { wave_real(K * R, K * v3) }
}

// Phi, dPhi/dxb, dPhi/dvb and the second derivatives d2/dxb2, d2/dxb dvb, d2/dvb2
// from the derivatives of the tile polynomials of dPhi/dxb and dPhi/dvb.
#[inline]
fn phi_tiles2(fd: &FiniteDepth, which: i32, xb: f64, vb: f64) -> [f64; 6] {
    const N: usize = TN;
    let (c, tx, ty, ny) = tile_of(fd, which, xb, vb);
    let ymax = if which == 1 { 1.0 } else { 2.0 };
    let sx = 2.0 * FiniteDepth::TS as f64 / XMAX; // dtx/dxb
    let sy = 2.0 * ny as f64 / ymax; // dty/dvb
    let mut v = [0.0f64; 3];
    let mut vx = [0.0f64; 3];
    let mut vy = [0.0f64; 3];
    for i in (0..N).rev() {
        for q in 0..3 {
            let r = &c[q * N * N + i * N..q * N * N + i * N + N];
            let mut p = r[N - 1];
            let mut dp = 0.0;
            for j in (0..N - 1).rev() {
                dp = dp * ty + p;
                p = p * ty + r[j];
            }
            vx[q] = vx[q] * tx + v[q];
            v[q] = v[q] * tx + p;
            vy[q] = vy[q] * tx + dp;
        }
    }
    [v[0], v[1], v[2], sx * vx[1], sy * vy[1], sy * vy[2]]
}

static TABLES_OK: LazyLock<bool> = LazyLock::new(|| {
    let mut h: u64 = 0xCBF29CE484222325;
    h = fnv1a(&T::REGION_META, h);
    h = fnv1a(&T::Q, h);
    h = fnv1a(&T::L0, h);
    h = fnv1a(&T::BESSEL_K, h);
    h == T::CHECKSUM
});

/// Verifies the checksum of the generated tables; fails with ErrorKind::Internal.
pub fn verify_fin_tables() -> Result<()> {
    if !*TABLES_OK {
        return Err(Error::internal("finite-depth Green function tables fail their checksum"));
    }
    Ok(())
}

/// Per-frequency data of the finite-depth Green function: depth h > 0, K = w^2/g >= 0 for the
/// ordinary frequencies, mode Ordinary or Zero / Infinite for the limits; tiles false keeps the
/// direct evaluation of the integral form (tests; the solves use the tiles).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn make_finite_depth(h: f64, K: f64, mode: FreqMode, tiles: bool) -> FiniteDepth {
    let mut fd = FiniteDepth { mode, h, sign: [1.0; 6], ..Default::default() };
    let mut cw = vec![0.0f64; (M + 1) * NC];
    fd.kn.resize(ROOTS as usize, 0.0);
    fd.cn.resize(ROOTS as usize, 0.0);
    if mode == FreqMode::Zero {
        for n in 1..=ROOTS {
            fd.kn[n as usize - 1] = n as f64 * PI / h;
            fd.cn[n as usize - 1] = 4.0 / h;
        }
        cw.copy_from_slice(&T::L0[..(M + 1) * NC]);
        power_table(&cw, &mut fd.P);
        if tiles {
            build_tiles(&mut fd);
        }
        return fd;
    }
    if mode == FreqMode::Infinite {
        fd.sign = [1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        for n in 1..=ROOTS {
            fd.kn[n as usize - 1] = (n as f64 - 0.5) * PI / h;
            fd.cn[n as usize - 1] = 4.0 / h;
        }
        region_coefficients(T::REGIONS - 1, 0.0, &mut cw); // t = 1/H = 0
        power_table(&cw, &mut fd.P);
        if tiles {
            build_tiles(&mut fd);
        }
        return fd;
    }
    assert!(K > 0.0, "finite depth: an ordinary frequency needs K > 0 (omega = 0 is FreqMode::Zero)");
    fd.K = K;
    fd.H = K * h;
    let x0 = dispersion_x0(fd.H);
    fd.k0 = x0 / h;
    fd.A0 = 2.0 * PI * fd.k0 / (-(-4.0 * x0).exp_m1() + 4.0 * x0 * (-2.0 * x0).exp());
    for n in 1..=ROOTS {
        let k = dispersion_xn(fd.H, n) / h;
        fd.kn[n as usize - 1] = k;
        fd.cn[n as usize - 1] = 4.0 * (k * k + K * K) / (k * k * h + K * K * h - K);
    }
    if fd.H < T::H_MIN {
        cw.copy_from_slice(&T::L0[..(M + 1) * NC]);
        // Q_0 = l_0 - log(H)/2 + O(H log H); the constant Chebyshev term is the constant power term
        cw[0] -= 0.5 * fd.H.ln();
    } else if fd.H >= T::HC {
        region_coefficients(T::REGIONS - 1, 1.0 / fd.H, &mut cw);
    } else {
        let tau = fd.H.ln();
        let mut r: i32 = 0;
        while r + 2 < T::REGIONS && tau > T::REGION_META[5 * r as usize + 2] {
            r += 1;
        }
        region_coefficients(r, tau, &mut cw);
    }
    power_table(&cw, &mut fd.P);
    if tiles {
        build_tiles(&mut fd);
    }
    fd
}

/// The fin_point data of the depth z (see FinPoint).
pub fn fin_point(fd: &FiniteDepth, z: f64) -> FinPoint {
    let mut p = FinPoint::default();
    let h = fd.h;
    let k0 = fd.k0;
    if fd.mode == FreqMode::Ordinary {
        p.d = (k0 * z).exp();
        p.a = (-k0 * (z + h)).exp();
        p.b = (-k0 * (h - z)).exp();
        p.c = (-k0 * (z + 2.0 * h)).exp();
    }
    for n in 0..fd.kn.len() {
        p.cs[2 * n] = (fd.kn[n] * (z + h)).cos();
        p.cs[2 * n + 1] = (fd.kn[n] * (z + h)).sin();
    }
    p
}

/// The eigenfunction series sum_n c_n cos k_n(z + h) cos k_n(zeta + h) K0(k_n R)
/// (the evanescent part of Re G at ordinary frequencies and at omega = 0, the whole G at
/// omega = inf) with its derivatives in R and zeta, from the fin_point data of z and zeta
/// (channel images). Valid for k_1 R >= 1/2, where the tiles of the scaled K0 and K1
/// begin: the channel rule W >= h / pi (route.rs) keeps every image there.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn finite_series(fd: &FiniteDepth, R: f64, pz: &FinPoint, pzeta: &FinPoint) -> (f64, f64, f64) {
    let pts = Some((pz, pzeta));
    let s = series_terms::<false>(
        fd,
        R,
        |_, x| scaled_k01(x),
        |n, k| series_depths(pts, n, k, 0.0, 0.0, fd.h),
        &mut KernHess::default(),
    );
    (s[0], s[1], s[2])
}

/// The second derivatives of finite_series for the sheet rows of a channel: with
/// Z_n = cos k_n(z + h), Z_n' = -k_n sin k_n(z + h) and K0'' = K0 + K1 / x,
/// (f_R, f_z, f_RR, f_Rz, f_Rzeta, f_zzeta) of f = sum_n c_n Z_n(z) Z_n(zeta) K0(k_n R).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn finite_series_hess(fd: &FiniteDepth, R: f64, pz: &FinPoint, pzeta: &FinPoint) -> [f64; 6] {
    let mut v = [0.0; 6];
    for n in 0..fd.kn.len() {
        let k = fd.kn[n];
        let x = k * R;
        if x >= SERIES_STOP {
            break;
        }
        let (g0, g1) = scaled_k01(x);
        let e = (-x).exp() / x.sqrt();
        let K0 = g0 * e;
        let K1 = g1 * e;
        let (cz, sz) = (pz.cs[2 * n], pz.cs[2 * n + 1]);
        let (cs, ss) = (pzeta.cs[2 * n], pzeta.cs[2 * n + 1]);
        let c = fd.cn[n];
        v[0] -= c * cz * cs * k * K1;
        v[1] -= c * k * sz * cs * K0;
        v[2] += c * cz * cs * k * k * (K0 + K1 / x);
        v[3] += c * sz * cs * k * k * K1;
        v[4] += c * cz * ss * k * k * K1;
        v[5] += c * k * k * sz * ss * K0;
    }
    v
}

/// Regular part Wr, Wi of the finite-depth Green function and its derivatives d/dR, d/dzeta,
/// d/dz: fd from make_finite_depth(), horizontal distance R >= 0, field-point depth z and source
/// depth zeta in [-h, 0], (R, z + zeta) != (0, 0) but for coincident points on z = 0 (the limit
/// of the integral form); pz, pzeta optional fin_point() data of z and zeta (the same results up
/// to round-off).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn finite_kernel(
    fd: &FiniteDepth,
    R: f64,
    z: f64,
    zeta: f64,
    pz: Option<&FinPoint>,
    pzeta: Option<&FinPoint>,
) -> Kern {
    let h = fd.h;
    let (sg, v) = vertical_distances(z, zeta, h);
    let ordinary = fd.mode == FreqMode::Ordinary;
    let pts = match (pz, pzeta) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    };
    // propagating mode: E = sum_{p=3..6} exp(-k0 v_p) and its derivatives
    let e = pair_depth_factor(fd, &v, pts);
    if R >= 0.5 * h {
        let jy = if ordinary { bessel_jy01(fd.k0 * R) } else { (0.0, 0.0, 0.0, 0.0) };
        return series_form::<false>(
            fd,
            R,
            &v,
            sg,
            &e,
            jy,
            |_, x| scaled_k01(x),
            |n, k| series_depths(pts, n, k, z, zeta, h),
            &mut KernHess::default(),
        );
    }
    let mut o = Kern::default();
    if fd.tiled {
        // integral form from the per-frequency tiles
        let ih = 1.0 / h;
        let ih2 = ih * ih;
        let f1 = phi_tiles(fd, 1, R * ih, v[0] * ih);
        let f2 = phi_tiles(fd, 2, R * ih, v[1] * ih);
        o.Wr = ih * (f1[0] + f2[0]);
        o.WrR = ih2 * (f1[1] + f2[1]);
        let A1z = -sg * ih2 * f1[2]; // d vb1 / d zeta = -sign(z - zeta) / h
        let mut Bz = ih2 * f2[2];
        if ordinary {
            let K = fd.K;
            let (F3, F3X, F3Y) = wave_real_or_limit(K, R, v[2]);
            o.Wr += K * F3;
            o.WrR += K * K * F3X;
            Bz -= K * K * F3Y;
            let (j0, j1) = bessel_j01(fd.k0 * R);
            imaginary_part(fd, &e, j0, j1, &mut o);
        }
        o.WrZ = A1z + Bz;
        o.Wrz = -A1z + Bz;
        return o;
    }
    // integral form (Newman's decomposition), direct evaluation (the reference of the tiles)
    let mut A1z = 0.0; // V1-dependent and symmetric parts of d Wr / d zeta
    let mut Bz = 0.0;
    if ordinary {
        let K = fd.K;
        let H = fd.H;
        let X = K * R;
        let V1 = K * v[0];
        let V2 = K * v[1];
        let (F3, F3X, F3Y) = wave_real_or_limit(K, R, v[2]);
        let (Fa, FaX, FaY) = wave_real(X, 2.0 * H - V1);
        let (Fb, FbX, FbY) = wave_real(X, 2.0 * H + V1);
        let (Fc, FcX, FcY) = wave_real(X, 2.0 * H + V2);
        let K2 = K * K;
        o.Wr = K * (F3 + Fa + Fb + Fc);
        o.WrR = K2 * (F3X + FaX + FbX + FcX);
        A1z = K2 * sg * (FaY - FbY);
        Bz = K2 * (FcY - F3Y);
        let (j0, j1) = bessel_j01(fd.k0 * R);
        imaginary_part(fd, &e, j0, j1, &mut o);
    }
    let ih = 1.0 / h;
    let s = R * R * ih * ih;
    let d1 = z - zeta;
    let (q, qs, qw) = q_eval2(&fd.P, s, d1 * d1 * ih * ih, v[1] * v[1] * ih * ih);
    let ih3 = ih * ih * ih;
    o.Wr += ih * (q[0] + q[1]);
    o.WrR += 2.0 * R * ih3 * (qs[0] + qs[1]);
    A1z += -2.0 * d1 * ih3 * qw[0];
    Bz += 2.0 * v[1] * ih3 * qw[1];
    o.WrZ = A1z + Bz;
    o.Wrz = -A1z + Bz;
    o
}

/// The regular part with its second derivatives (thin sheets in finite depth).
/// Near region from the derivatives of the tile polynomials (fd must be tiled) and the
/// Ft identities, far region from the series and Bessel recurrences; same arguments as
/// finite_kernel(), R > 0 or z + zeta < 0.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn finite_kernel_hess(
    fd: &FiniteDepth,
    R: f64,
    z: f64,
    zeta: f64,
    pz: Option<&FinPoint>,
    pzeta: Option<&FinPoint>,
) -> (Kern, KernHess) {
    // the value, first and second derivatives in one pass (the terms of finite_kernel
    // evaluated once)
    let mut hs = KernHess::default();
    let h = fd.h;
    let (sg, v) = vertical_distances(z, zeta, h);
    let ordinary = fd.mode == FreqMode::Ordinary;
    let pts = match (pz, pzeta) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    };
    // propagating depth factor E = sum exp(-k0 v_p), p = 3..6, and its derivatives
    let e = pair_depth_factor(fd, &v, pts);
    if R >= 0.5 * h {
        let jy = if ordinary { bessel_jy01(fd.k0 * R) } else { (0.0, 0.0, 0.0, 0.0) };
        let o = series_form::<true>(
            fd,
            R,
            &v,
            sg,
            &e,
            jy,
            |_, x| scaled_k01(x),
            |n, k| series_depths(pts, n, k, z, zeta, h),
            &mut hs,
        );
        return (o, hs);
    }
    assert!(fd.tiled, "finite_kernel_hess needs the tiled kernel");
    let mut o = Kern::default();
    let ih = 1.0 / h;
    let ih2 = ih * ih;
    let ih3 = ih2 * ih;
    let xb = R * ih;
    let a = phi_tiles2(fd, 1, xb, v[0] * ih);
    let b = phi_tiles2(fd, 2, xb, v[1] * ih);
    o.Wr = ih * (a[0] + b[0]);
    o.WrR = ih2 * (a[1] + b[1]);
    let A1z = -sg * ih2 * a[2]; // d vb1 / d zeta = -sign(z - zeta) / h
    let mut Bz = ih2 * b[2];
    hs.RR = ih3 * (a[3] + b[3]);
    // Phi is even in xb: dPhi/dxb / xb, near the axis less the tiles' own derivative there (about
    // 2e-7 instead of 0), which a division by a small xb would amplify
    if xb >= AXIS_X {
        hs.R_R = ih3 * (a[1] + b[1]) / xb;
    } else if xb > 1e-12 {
        let a0 = phi_tiles2(fd, 1, 0.0, v[0] * ih);
        let b0 = phi_tiles2(fd, 2, 0.0, v[1] * ih);
        hs.R_R = ih3 * ((a[1] - a0[1]) + (b[1] - b0[1])) / xb;
    } else {
        hs.R_R = hs.RR;
    }
    hs.RZ = ih3 * (-sg * a[4] + b[4]);
    hs.Rz = ih3 * (sg * a[4] + b[4]);
    hs.zZ = ih3 * (-a[5] + b[5]);
    if ordinary {
        // K Ft(X, Y), X = K R, Y = K v3 (dY/dzeta = dY/dz = -K): Ft_Y = -Ft - 2/R0 and
        // Laplace's equation give the second derivatives (as sheet_wave)
        let K = fd.K;
        let K3 = K * K * K;
        let X = K * R;
        let Y = K * v[2];
        if R == 0.0 && v[2] == 0.0 {
            // coincident points on z = 0: the limit of Ft + 2 psi; no second derivatives
            o.Wr += -2.0 * K * (GAMMA - LOG2 + K.ln());
        } else {
            let (F, FX, FY) = wave_real(X, Y);
            o.Wr += K * F;
            o.WrR += K * K * FX;
            Bz -= K * K * FY;
            let R0 = (X * X + Y * Y).sqrt();
            let iR03 = 1.0 / (R0 * R0 * R0);
            let FYY = -FY + 2.0 * Y * iR03;
            let FXY = -FX + 2.0 * X * iR03;
            let FX_X = wave_fx_over_x(X, Y, FX, FYY); // accurate near the axis
            let FXX = -FX_X - FYY;
            hs.RR += K3 * FXX;
            hs.R_R += K3 * FX_X;
            hs.RZ -= K3 * FXY;
            hs.Rz -= K3 * FXY;
            hs.zZ += K3 * FYY;
        }
        let (j0, j1) = bessel_j01(fd.k0 * R);
        imaginary_part(fd, &e, j0, j1, &mut o);
        imaginary_hess(fd, &e, R, j0, j1, &mut hs);
    }
    o.WrZ = A1z + Bz;
    o.Wrz = -A1z + Bz;
    (o, hs)
}
