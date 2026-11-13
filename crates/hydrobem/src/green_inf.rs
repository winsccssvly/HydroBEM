//! The infinite-depth wave kernel (theory §8): the tables of scripts/gen_green_tables.py,
//! the asymptotic and series forms (Newman 1985), Bessel J and Y.
//!
//! Infinite-depth free-surface Green function, wave part:
//!
//!   G = 1/r + 1/r' + K Ft(X, Y) + i K Gi(X, Y),   X = K R, Y = K |z + zeta|,
//!   Gi = -2 pi exp(-Y) J0(X)   (time factor exp(i w t), Newman 1985 eq. 2).
//!
//! Why Chebyshev tiles of these degrees, the identity form inside [0, 2)^2 and the direct tables
//! outside it: theory §8 and §8.1 (each degree chosen on the error map against 2e-6);
//! why four points at a time without fused multiply-add: theory §8.3 (grounds).

use std::sync::LazyLock;

use crate::constants::{GAMMA, LOG2, PI, fnv1a};
use crate::env_hooks::diag_double;
use crate::error::{Error, Result};
use crate::tables::green_inf_tables as T;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct WaveKernel {
    pub F: f64, // Ft and its X, Y derivatives
    pub FX: f64,
    pub FY: f64,
    pub I: f64, // Gi and its X, Y derivatives
    pub IX: f64,
    pub IY: f64,
    pub c: f64, // exp(-Y) J0(X): weight of the log singularity
}

/// Below this X the quotients F_X / X of the second derivatives subtract the value of F_X on the
/// axis X = 0 first: the tables give F_X(0, Y) only to their accuracy (about 3e-8), which a
/// division by a small X would amplify.
pub const AXIS_X: f64 = 0.1;

const XT: i32 = T::XT;
const YT: i32 = T::YT;
const XTF: f64 = T::XT as f64;
const YTF: f64 = T::YT as f64;
const N1: usize = T::N1 as usize;
const N2: usize = T::N2 as usize;
const N3: usize = T::N3 as usize;
const HANKEL_TERMS: usize = T::HANKEL_TERMS as usize;
const STRUVE_TERMS: usize = T::STRUVE_TERMS as usize;

const GAMMA_MINUS_LOG2: f64 = GAMMA - LOG2;
// terms m = 0..14 of the partial-integration series: the series stops when the bound
// m! / A^((m+1)/2) of the next term is below 1e-14, or after 15 terms (at the table edge r = 30
// the bound of the last term left is about 6e-12, within the 2e-6 of theory §8; the measured
// error of the far field, theory §8.1, is far below that bound)
const FAR_TERMS: i32 = 15;

// 1D tables: [tile][power][function], functions J0, J1, Yt, Yt'; Horner in
// the tile variable t in [-1, 1] for the four functions at once.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn table1d(X: f64) -> [f64; 4] {
    let mut i = X as i32;
    if i >= XT {
        i = XT - 1;
    }
    let t = 2.0 * (X - i as f64) - 1.0;
    let off = i as usize * (N1 + 1) * 4;
    let c: &[f64; (N1 + 1) * 4] = T::TABLE_1D[off..off + (N1 + 1) * 4].try_into().unwrap();
    let mut r = [0.0f64; 4];
    for j in (0..=N1).rev() {
        for f in 0..4 {
            r[f] = r[f] * t + c[4 * j + f];
        }
    }
    r
}

// 2D tables: [iy][ix][function][x power][y power], functions V, V_X: (V, V_X).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn table2d(X: f64, Y: f64) -> (f64, f64) {
    let mut ix = X as i32;
    let mut iy = Y as i32;
    if ix >= XT {
        ix = XT - 1;
    }
    if iy >= YT {
        iy = YT - 1;
    }
    let tx = 2.0 * (X - ix as f64) - 1.0;
    let ty = 2.0 * (Y - iy as f64) - 1.0;
    const M: usize = N2 + 1;
    let off = (iy as usize * XT as usize + ix as usize) * 2 * M * M;
    let c: &[f64; 2 * M * M] = T::TABLE_2D[off..off + 2 * M * M].try_into().unwrap();
    // per function the Horner chains in tx, then in ty
    let mut s = [0.0f64; 2];
    for q in 0..2 {
        let cq: &[f64; M * M] = c[q * M * M..(q + 1) * M * M].try_into().unwrap();
        let mut r = [0.0f64; M];
        for j in (0..M).rev() {
            for a in 0..M {
                r[a] = r[a] * tx + cq[j * M + a];
            }
        }
        for a in (0..M).rev() {
            s[q] = s[q] * ty + r[a];
        }
    }
    (s[0], s[1])
}

// Direct 2D tables: [iy][ix][function][x power][y power],
// functions Ft, Ft_X, exp(-Y) J0, exp(-Y) J1, valid outside [0, 2) x [0, 2).
// NQ = 2: only Ft and Ft_X (a const generic).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn table2d_direct<const NQ: usize>(X: f64, Y: f64) -> [f64; NQ] {
    let mut f = [0.0f64; NQ];
    let ix = X as i32;
    let iy = Y as i32;
    let tx = 2.0 * (X - ix as f64) - 1.0;
    let ty = 2.0 * (Y - iy as f64) - 1.0;
    const M: usize = N3 + 1;
    let off = (iy as usize * XT as usize + ix as usize) * 4 * M * M;
    let c: &[f64; 4 * M * M] = T::TABLE_2D_DIRECT[off..off + 4 * M * M].try_into().unwrap();
    // per function q the Horner chains in tx (one per y power a), then in ty
    for q in 0..NQ {
        let cq: &[f64; M * M] = c[q * M * M..(q + 1) * M * M].try_into().unwrap();
        let mut r = [0.0f64; M];
        for j in (0..M).rev() {
            for a in 0..M {
                r[a] = r[a] * tx + cq[j * M + a];
            }
        }
        let mut s = 0.0;
        for a in (0..M).rev() {
            s = s * ty + r[a];
        }
        f[q] = s;
    }
    f
}

// Hankel asymptotics for J0, Y0, J1, Y1, x >= XT (A&S 9.2.5-10); the phases
// chi_1 = chi_0 - pi/2 share one sine and cosine. Returns (J0, Y0, J1, Y1).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn hankel01(x: f64) -> (f64, f64, f64, f64) {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let mut P = [0.0f64; 2];
    let mut Q = [0.0f64; 2];
    for nu in 0..2 {
        let a = &T::ASYMPTOTIC[nu * HANKEL_TERMS..(nu + 1) * HANKEL_TERMS];
        let mut pw = 1.0;
        let mut k = 0;
        while 2 * k + 1 < HANKEL_TERMS {
            let s = if k % 2 != 0 { -1.0 } else { 1.0 };
            P[nu] += s * a[2 * k] * pw;
            Q[nu] += s * a[2 * k + 1] * pw * inv;
            pw *= inv2;
            k += 1;
        }
    }
    let chi = x - 0.25 * PI;
    let sq = (2.0 / (PI * x)).sqrt();
    let c = chi.cos();
    let sn = chi.sin();
    let j0 = sq * (P[0] * c - Q[0] * sn);
    let y0 = sq * (P[0] * sn + Q[0] * c);
    // cos(chi - pi/2) = sin(chi), sin(chi - pi/2) = -cos(chi)
    let j1 = sq * (P[1] * sn + Q[1] * c);
    let y1 = sq * (-P[1] * c + Q[1] * sn);
    (j0, y0, j1, y1)
}

// H_nu - Y_nu asymptotics (nu = 0, 1), x >= XT.
#[inline]
fn struve_minus_y(nu: usize, x: f64) -> f64 {
    let off = 2 * HANKEL_TERMS + nu * STRUVE_TERMS;
    let c = &T::ASYMPTOTIC[off..off + STRUVE_TERMS];
    let inv2 = 1.0 / (x * x);
    let mut s = 0.0;
    let mut pw = if nu == 0 { 1.0 / x } else { 1.0 };
    for &ck in c {
        s += ck * pw;
        pw *= inv2;
    }
    s
}

// Partial integration of Havelock's integral (Newman 1985 eq. 7):
//   int_0^Y exp(t-Y) f(t) dt = sum_m (-1)^m [f^(m)(Y) - exp(-Y) f^(m)(0)] + R_M,
// f(t) = (X^2 + t^2)^(-1/2), |R_M| <= max |f^(M+1)| <= (M+1)! / X^(M+2).
// The Taylor coefficients a_n = f^(n)(t0)/n! follow from (X^2 + t^2) f' + t f = 0:
//   a_{n+1} = -[(2n+1) t0 a_n + n a_{n-1}] / (A (n+1)),  A = X^2 + t0^2,
// and b_n = d a_n / dX by differentiating the recurrence. The sum stops when
// m! / A^((m+1)/2) < FAR_TOL.
const FAR_TOL: f64 = 1e-14;
// 1/n for the recurrence, n = 1..FAR_TERMS - 1 (index 0 unused)
const INV_N: [f64; FAR_TERMS as usize] = [
    0.0,
    1.0,
    1.0 / 2.0,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 5.0,
    1.0 / 6.0,
    1.0 / 7.0,
    1.0 / 8.0,
    1.0 / 9.0,
    1.0 / 10.0,
    1.0 / 11.0,
    1.0 / 12.0,
    1.0 / 13.0,
    1.0 / 14.0,
];

// sum_m (-1)^m m! a_m(t0) and the same for b_m, with the adaptive stop: (P, P_X).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn taylor_sum(X: f64, t0: f64) -> (f64, f64) {
    let A = X * X + t0 * t0;
    let ia = 1.0 / A;
    let ir = ia.sqrt();
    let mut a = ir; // a_0
    let mut b = -X * ir * ia; // b_0
    let mut am1 = 0.0;
    let mut bm1 = 0.0;
    let mut fact = 1.0; // m!
    let mut size = ir; // m! / A^((m+1)/2)
    let mut P = a;
    let mut PX = b;
    let mut n = 0;
    while n + 1 < FAR_TERMS {
        let an = a;
        let bn = b;
        let nf = n as f64;
        let k = (2 * n + 1) as f64;
        a = -(k * t0 * an + nf * am1) * ia * INV_N[(n + 1) as usize];
        b = -(k * t0 * bn + nf * bm1) * ia * INV_N[(n + 1) as usize] - a * 2.0 * X * ia;
        am1 = an;
        bm1 = bn;
        fact *= (n + 1) as f64;
        size *= (n + 1) as f64 * ir;
        let s = if n % 2 != 0 { fact } else { -fact }; // (-1)^(n+1) (n+1)!
        P += s * a;
        PX += s * b;
        if size < FAR_TOL {
            break;
        }
        n += 1;
    }
    (P, PX)
}

// (P, P_X) of Havelock's integral with the weight w = exp(-Y) of its lower end.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn far_series(X: f64, Y: f64, w: f64) -> (f64, f64) {
    let (mut P, mut PX) = taylor_sum(X, Y);
    if w > 1e-17 {
        // exp(-Y) part; negligible beyond Y ~ 39
        let (P0, PX0) = taylor_sum(X, 0.0);
        P -= w * P0;
        PX -= w * PX0;
    }
    (P, PX)
}

static TABLES_OK: LazyLock<bool> = LazyLock::new(|| {
    let mut h: u64 = 0xCBF29CE484222325;
    h = fnv1a(&T::TABLE_1D, h);
    h = fnv1a(&T::TABLE_2D, h);
    h = fnv1a(&T::TABLE_2D_DIRECT, h);
    h = fnv1a(&T::ASYMPTOTIC, h);
    h == T::CHECKSUM
});

/// Verifies the checksum of the generated tables; fails with ErrorKind::Internal.
pub fn verify_green_tables() -> Result<()> {
    if !*TABLES_OK {
        return Err(Error::internal("infinite-depth Green function tables fail their checksum"));
    }
    Ok(())
}

/// Bessel functions (J0, J1) for X >= 0 (tables and Hankel asymptotics).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn bessel_j01(X: f64) -> (f64, f64) {
    if X < XTF {
        let f = table1d(X);
        (f[0], f[1])
    } else {
        let (j0, _, j1, _) = hankel01(X);
        (j0, j1)
    }
}

/// Bessel functions (J0, Y0, J1, Y1) for X > 0.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn bessel_jy01(X: f64) -> (f64, f64, f64, f64) {
    if X >= XTF {
        return hankel01(X);
    }
    // Y0 = Yt + (2/pi) L J0 and Y1 = -Y0' with L = log(X/2) + gamma (green_reference.py)
    let f = table1d(X);
    let j0 = f[0];
    let j1 = f[1];
    let L = (0.5 * X).ln() + GAMMA;
    let y0 = f[2] + 2.0 / PI * L * j0;
    let y1 = -f[3] - 2.0 / PI * (j0 / X - L * j1);
    (j0, y0, j1, y1)
}

/// Wave part Ft, Gi and their derivatives of the infinite-depth Green function at
/// X = K R >= 0, Y = K |z + zeta| >= 0, (X, Y) != (0, 0): values and derivatives in X and Y.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_kernel(X: f64, Y: f64) -> WaveKernel {
    let mut out = WaveKernel::default();
    let R0 = (X * X + Y * Y).sqrt();
    if !diag_double() && X < XTF && Y < YTF && (X >= 2.0 || Y >= 2.0) {
        let f = table2d_direct::<4>(X, Y);
        out.F = f[0];
        out.FX = f[1];
        out.FY = -f[0] - 2.0 / R0;
        out.c = f[2];
        out.I = -2.0 * PI * f[2];
        out.IX = 2.0 * PI * f[3];
        out.IY = -out.I;
        return out;
    }
    let eY = (-Y).exp();
    let j0;
    let j1;
    if X < XTF && Y < YTF {
        let f = table1d(X);
        j0 = f[0];
        j1 = f[1];
        let yt = f[2];
        let dyt = f[3];
        let (v, vx) = table2d(X, Y);
        let lg = (Y + R0).ln() + GAMMA_MINUS_LOG2;
        out.F = -2.0 * eY * j0 * lg - PI * eY * yt - 2.0 * R0 * v;
        out.FX = 2.0 * eY * j1 * lg
            - 2.0 * eY * j0 * X / (R0 * (Y + R0))
            - PI * eY * dyt
            - 2.0 * (X / R0) * v
            - 2.0 * R0 * vx;
    } else if Y >= YTF {
        // exp(-Y) terms are below 1e-13 relative and are dropped
        (j0, j1) = bessel_j01(X);
        let (P, PX) = far_series(X, Y, 0.0);
        out.F = -2.0 * P;
        out.FX = -2.0 * PX;
    } else {
        let (y0, y1);
        (j0, y0, j1, y1) = hankel01(X);
        let h0py0 = struve_minus_y(0, X) + 2.0 * y0; // H0 + Y0
        let dh0py0 = 2.0 / PI - struve_minus_y(1, X) - 2.0 * y1; // (H0 + Y0)'
        let (P, PX) = far_series(X, Y, eY);
        out.F = -PI * eY * h0py0 - 2.0 * P;
        out.FX = -PI * eY * dh0py0 - 2.0 * PX;
    }
    out.FY = -out.F - 2.0 / R0; // Ft_Y = -Ft - 2/R0 (from eq. 3a)
    out.c = eY * j0;
    out.I = -2.0 * PI * eY * j0;
    out.IX = 2.0 * PI * eY * j1;
    out.IY = -out.I;
    out
}

/// Whether wave_kernel_near4 holds the four points: all in the near branch (X < 2, Y < 2, not
/// both zero) and in one table tile (the same integer parts of X and of Y).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn near4_applies(X: &[f64; 4], Y: &[f64; 4]) -> bool {
    let (ix, iy) = (X[0] as i32, Y[0] as i32);
    (0..4).all(|k| X[k] < 2.0 && Y[k] < 2.0 && X[k] as i32 == ix && Y[k] as i32 == iy && X[k] + Y[k] > 0.0)
}

/// wave_kernel at four points of one tile of the near branch (near4_applies), with AVX2: the
/// Horner chains of table1d and table2d run in the four lanes on the coefficients of the one
/// tile; exp(-Y) and log(Y + R0) per point. Every operation is the scalar one in the same order
/// (no fused multiply-add), so the result equals wave_kernel's to the bit.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_kernel_near4(X: &[f64; 4], Y: &[f64; 4]) -> [WaveKernel; 4] {
    use std::arch::x86_64::*;
    debug_assert!(near4_applies(X, Y));
    let R0: [f64; 4] = std::array::from_fn(|k| (X[k] * X[k] + Y[k] * Y[k]).sqrt());
    let eY: [f64; 4] = std::array::from_fn(|k| (-Y[k]).exp());
    let lg: [f64; 4] = std::array::from_fn(|k| (Y[k] + R0[k]).ln() + GAMMA_MINUS_LOG2);
    let (ix, iy) = (X[0] as i32 as usize, Y[0] as i32 as usize);
    const M: usize = N2 + 1;
    let c1: &[f64; (N1 + 1) * 4] = T::TABLE_1D[ix * (N1 + 1) * 4..(ix + 1) * (N1 + 1) * 4].try_into().unwrap();
    let o2 = (iy * XT as usize + ix) * 2 * M * M;
    let c2: &[f64; 2 * M * M] = T::TABLE_2D[o2..o2 + 2 * M * M].try_into().unwrap();
    let mut j = [[0.0f64; 4]; 4]; // J0, J1, Yt, Yt' per lane
    let mut f = [0.0f64; 4];
    let mut fx = [0.0f64; 4];
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); the loads and stores read
    // and write the arrays of four doubles above.
    unsafe {
        let b = |v: f64| _mm256_set1_pd(v);
        let ld = |a: &[f64; 4]| _mm256_loadu_pd(a.as_ptr());
        let (x, y) = (ld(X), ld(Y));
        // the tile variables 2 (X - i) - 1 of table1d and table2d (one tile: tx of table2d is t)
        let t = _mm256_sub_pd(_mm256_mul_pd(b(2.0), _mm256_sub_pd(x, b(ix as f64))), b(1.0));
        let ty = _mm256_sub_pd(_mm256_mul_pd(b(2.0), _mm256_sub_pd(y, b(iy as f64))), b(1.0));
        let mut r = [_mm256_setzero_pd(); 4];
        for jj in (0..=N1).rev() {
            for q in 0..4 {
                r[q] = _mm256_add_pd(_mm256_mul_pd(r[q], t), b(c1[4 * jj + q]));
            }
        }
        let mut s = [_mm256_setzero_pd(); 2];
        for q in 0..2 {
            let mut rr = [_mm256_setzero_pd(); M];
            for jj in (0..M).rev() {
                for a in 0..M {
                    rr[a] = _mm256_add_pd(_mm256_mul_pd(rr[a], t), b(c2[q * M * M + jj * M + a]));
                }
            }
            for a in (0..M).rev() {
                s[q] = _mm256_add_pd(_mm256_mul_pd(s[q], ty), rr[a]);
            }
        }
        let (j0, j1, yt, dyt, v, vx) = (r[0], r[1], r[2], r[3], s[0], s[1]);
        let (e, l, r0) = (ld(&eY), ld(&lg), ld(&R0));
        let m = |a: __m256d, c: __m256d| _mm256_mul_pd(a, c);
        let sub = |a: __m256d, c: __m256d| _mm256_sub_pd(a, c);
        let two = b(2.0);
        // F = -2 eY J0 lg - pi eY Yt - 2 R0 V
        let F = sub(sub(m(m(m(b(-2.0), e), j0), l), m(m(b(PI), e), yt)), m(m(two, r0), v));
        // F_X = 2 eY J1 lg - 2 eY J0 X / (R0 (Y + R0)) - pi eY Yt' - 2 (X / R0) V - 2 R0 V_X
        let a1 = m(m(m(two, e), j1), l);
        let a2 = _mm256_div_pd(m(m(m(two, e), j0), x), m(r0, _mm256_add_pd(y, r0)));
        let a3 = m(m(b(PI), e), dyt);
        let a4 = m(m(two, _mm256_div_pd(x, r0)), v);
        let a5 = m(m(two, r0), vx);
        let FX = sub(sub(sub(sub(a1, a2), a3), a4), a5);
        for (q, v) in [j0, j1, yt, dyt].into_iter().enumerate() {
            _mm256_storeu_pd(j[q].as_mut_ptr(), v);
        }
        _mm256_storeu_pd(f.as_mut_ptr(), F);
        _mm256_storeu_pd(fx.as_mut_ptr(), FX);
    }
    std::array::from_fn(|k| {
        let mut out = WaveKernel { F: f[k], FX: fx[k], ..Default::default() };
        out.FY = -out.F - 2.0 / R0[k];
        out.c = eY[k] * j[0][k];
        out.I = -2.0 * PI * eY[k] * j[0][k];
        out.IX = 2.0 * PI * eY[k] * j[1][k];
        out.IY = -out.I;
        out
    })
}

/// Whether wave_kernel_direct4 and wave_real_direct4 hold the four points: all in the branch of
/// the direct tables (X < XT, Y < YT, outside [0, 2)²) and in one table tile.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn direct4_applies(X: &[f64; 4], Y: &[f64; 4]) -> bool {
    let (ix, iy) = (X[0] as i32, Y[0] as i32);
    (0..4).all(|k| X[k] < XTF && Y[k] < YTF && (X[k] >= 2.0 || Y[k] >= 2.0) && X[k] as i32 == ix && Y[k] as i32 == iy)
        && !diag_double()
}

// table2d_direct at four points of one tile (direct4_applies): its Horner chains in the four
// lanes on the coefficients of that tile, every operation the scalar one in the same order.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn table2d_direct4<const NQ: usize>(X: &[f64; 4], Y: &[f64; 4]) -> [[f64; 4]; NQ] {
    use std::arch::x86_64::*;
    let (ix, iy) = (X[0] as i32, Y[0] as i32);
    const M: usize = N3 + 1;
    let off = (iy as usize * XT as usize + ix as usize) * 4 * M * M;
    let c: &[f64; 4 * M * M] = T::TABLE_2D_DIRECT[off..off + 4 * M * M].try_into().unwrap();
    let mut f = [[0.0f64; 4]; NQ];
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); the loads and stores read
    // and write arrays of four doubles.
    unsafe {
        let b = |v: f64| _mm256_set1_pd(v);
        let (x, y) = (_mm256_loadu_pd(X.as_ptr()), _mm256_loadu_pd(Y.as_ptr()));
        let tx = _mm256_sub_pd(_mm256_mul_pd(b(2.0), _mm256_sub_pd(x, b(ix as f64))), b(1.0));
        let ty = _mm256_sub_pd(_mm256_mul_pd(b(2.0), _mm256_sub_pd(y, b(iy as f64))), b(1.0));
        for q in 0..NQ {
            let cq = &c[q * M * M..(q + 1) * M * M];
            let mut r = [_mm256_setzero_pd(); M];
            for j in (0..M).rev() {
                for a in 0..M {
                    r[a] = _mm256_add_pd(_mm256_mul_pd(r[a], tx), b(cq[j * M + a]));
                }
            }
            let mut s = _mm256_setzero_pd();
            for a in (0..M).rev() {
                s = _mm256_add_pd(_mm256_mul_pd(s, ty), r[a]);
            }
            _mm256_storeu_pd(f[q].as_mut_ptr(), s);
        }
    }
    f
}

/// wave_kernel at four points of one tile of the direct tables (direct4_applies), with AVX2:
/// equal to wave_kernel's to the bit (the operations of its direct branch in the same order).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_kernel_direct4(X: &[f64; 4], Y: &[f64; 4]) -> [WaveKernel; 4] {
    debug_assert!(direct4_applies(X, Y));
    let f = table2d_direct4::<4>(X, Y);
    std::array::from_fn(|k| {
        let R0 = (X[k] * X[k] + Y[k] * Y[k]).sqrt();
        let mut out = WaveKernel { F: f[0][k], FX: f[1][k], ..Default::default() };
        out.FY = -f[0][k] - 2.0 / R0;
        out.c = f[2][k];
        out.I = -2.0 * PI * f[2][k];
        out.IX = 2.0 * PI * f[3][k];
        out.IY = -out.I;
        out
    })
}

/// wave_real at four points of one tile of the direct tables (direct4_applies), with AVX2: equal
/// to wave_real's to the bit.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_real_direct4(X: &[f64; 4], Y: &[f64; 4]) -> [(f64, f64, f64); 4] {
    debug_assert!(direct4_applies(X, Y));
    let f = table2d_direct4::<2>(X, Y);
    std::array::from_fn(|k| {
        let R0 = (X[k] * X[k] + Y[k] * Y[k]).sqrt();
        (f[0][k], f[1][k], -f[0][k] - 2.0 / R0)
    })
}

/// bessel_j01 at four points of one unit tile of its tables (X < XT, the same integer part):
/// the Horner chains of table1d for J0 and J1 in the four lanes (AVX2), every operation the scalar
/// one in the same order, so equal to bessel_j01's to the bit; None (nothing computed) otherwise.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn bessel_j01_4(X: &[f64; 4]) -> Option<([f64; 4], [f64; 4])> {
    use std::arch::x86_64::*;
    let i = X[0] as i32;
    if !(0..4).all(|k| X[k] < XTF && X[k] as i32 == i) {
        return None;
    }
    let c: &[f64; (N1 + 1) * 4] =
        T::TABLE_1D[i as usize * (N1 + 1) * 4..(i as usize + 1) * (N1 + 1) * 4].try_into().unwrap();
    let (mut j0, mut j1) = ([0.0f64; 4], [0.0f64; 4]);
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); the loads and stores read
    // and write arrays of four doubles.
    unsafe {
        let b = |v: f64| _mm256_set1_pd(v);
        let x = _mm256_loadu_pd(X.as_ptr());
        let t = _mm256_sub_pd(_mm256_mul_pd(b(2.0), _mm256_sub_pd(x, b(i as f64))), b(1.0));
        let (mut r0, mut r1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        for jj in (0..=N1).rev() {
            r0 = _mm256_add_pd(_mm256_mul_pd(r0, t), b(c[4 * jj]));
            r1 = _mm256_add_pd(_mm256_mul_pd(r1, t), b(c[4 * jj + 1]));
        }
        _mm256_storeu_pd(j0.as_mut_ptr(), r0);
        _mm256_storeu_pd(j1.as_mut_ptr(), r1);
    }
    Some((j0, j1))
}

/// bessel_jy01 at four points of one unit tile of its tables (0 < X < XT, the same integer part):
/// the Horner chains of table1d in the four lanes (AVX2), the logarithm and the rest per point,
/// every operation the scalar one in the same order, so equal to bessel_jy01's to the bit; None
/// (nothing computed) otherwise. Returns (J0, Y0, J1, Y1) per point.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn bessel_jy01_4(X: &[f64; 4]) -> Option<[(f64, f64, f64, f64); 4]> {
    use std::arch::x86_64::*;
    let i = X[0] as i32;
    if !(0..4).all(|k| X[k] > 0.0 && X[k] < XTF && X[k] as i32 == i) {
        return None;
    }
    let c: &[f64; (N1 + 1) * 4] =
        T::TABLE_1D[i as usize * (N1 + 1) * 4..(i as usize + 1) * (N1 + 1) * 4].try_into().unwrap();
    let mut f = [[0.0f64; 4]; 4]; // J0, J1, Yt, Yt' per lane
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); the loads and stores read
    // and write arrays of four doubles.
    unsafe {
        let b = |v: f64| _mm256_set1_pd(v);
        let x = _mm256_loadu_pd(X.as_ptr());
        let t = _mm256_sub_pd(_mm256_mul_pd(b(2.0), _mm256_sub_pd(x, b(i as f64))), b(1.0));
        let mut r = [_mm256_setzero_pd(); 4];
        for jj in (0..=N1).rev() {
            for q in 0..4 {
                r[q] = _mm256_add_pd(_mm256_mul_pd(r[q], t), b(c[4 * jj + q]));
            }
        }
        for q in 0..4 {
            _mm256_storeu_pd(f[q].as_mut_ptr(), r[q]);
        }
    }
    Some(std::array::from_fn(|k| {
        let (j0, j1) = (f[0][k], f[1][k]);
        let L = (0.5 * X[k]).ln() + GAMMA;
        let y0 = f[2][k] + 2.0 / PI * L * j0;
        let y1 = -f[3][k] - 2.0 / PI * (j0 / X[k] - L * j1);
        (j0, y0, j1, y1)
    }))
}

// F_X by the V tables (table1d and table2d, ~2e-10) where the direct tables of Y >= 2 hold the
// kernel: the direct tables are accurate to ~2e-8, but (F_X(X) - F_X(0)) / X near the axis takes
// the derivative of their error, ~3e-5 of the scale
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn fx_vtables(X: f64, Y: f64) -> f64 {
    let R0 = (X * X + Y * Y).sqrt();
    let eY = (-Y).exp();
    let f = table1d(X);
    let j0 = f[0];
    let j1 = f[1];
    let dyt = f[3];
    let (v, vx) = table2d(X, Y);
    let lg = (Y + R0).ln() + GAMMA_MINUS_LOG2;
    2.0 * eY * j1 * lg - 2.0 * eY * j0 * X / (R0 * (Y + R0)) - PI * eY * dyt - 2.0 * (X / R0) * v - 2.0 * R0 * vx
}

/// F_X / X of the real part of the wave kernel from F_X and F_YY at (X, Y), accurate near the axis
/// (below AXIS_X: the difference with the axis value through the V tables).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_fx_over_x(X: f64, Y: f64, FX: f64, FYY: f64) -> f64 {
    let direct = !diag_double() && (2.0..YTF).contains(&Y); // the direct tables hold F_X there (X < 2)
    if Y <= X {
        return FX / X; // near the singularity at (0, 0)
    }
    if X >= AXIS_X {
        return (if direct && X < 2.0 { fx_vtables(X, Y) } else { FX }) / X;
    }
    if X < 1e-12 {
        return -0.5 * FYY; // F_XX = F_X / X on the axis, F_XX + F_X / X + F_YY = 0
    }
    if Y < YTF {
        return (fx_vtables(X, Y) - fx_vtables(0.0, Y)) / X; // the tables' axis error cancels
    }
    // the far series (Y >= YT): accurate, its axis value 0 up to round-off
    let (_, FX0, _) = wave_real(0.0, Y);
    (FX - FX0) / X
}

/// (F_X / X, Gi_X / X) of the wave part at (X, Y) from its derivatives there (w, from wave_kernel):
/// (F_X(X) - F_X(0)) / X below AXIS_X (where Y > X), the limit on the axis (Laplace's and Bessel's
/// equations).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_x_over_x(X: f64, Y: f64, w: &WaveKernel, FYY: f64) -> (f64, f64) {
    let FX_X = wave_fx_over_x(X, Y, w.FX, FYY);
    let IX_X = if X >= AXIS_X || Y <= X {
        w.IX / X
    } else if X < 1e-12 {
        PI * (-Y).exp() // 2 pi e^{-Y} J1(X) / X -> pi e^{-Y}
    } else {
        let (_, j1) = bessel_j01(X);
        2.0 * PI * (-Y).exp() * j1 / X // Gi_X = 2 pi e^{-Y} J1(X), the quotient exact
    };
    (FX_X, IX_X)
}

/// Real part only: (Ft, dFt/dX, dFt/dY) (the same approximations as wave_kernel).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_real(X: f64, Y: f64) -> (f64, f64, f64) {
    let R0 = (X * X + Y * Y).sqrt();
    let (F, FX);
    if !diag_double() && X < XTF && Y < YTF && (X >= 2.0 || Y >= 2.0) {
        let f = table2d_direct::<2>(X, Y);
        F = f[0];
        FX = f[1];
    } else if Y >= YTF {
        let (P, PX) = far_series(X, Y, 0.0);
        F = -2.0 * P;
        FX = -2.0 * PX;
    } else {
        let w = wave_kernel(X, Y);
        F = w.F;
        FX = w.FX;
    }
    (F, FX, -F - 2.0 / R0)
}
