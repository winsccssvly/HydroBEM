//! The far-image tail of the channel Green function: the Faddeeva function by Weideman's
//! rational approximation and the periodic lattice sums (strip modes, Ewald); per-frequency tiles.
//!
//! Channel between the walls y = +-W/2 in finite depth:
//! the far-image (|m| >= 2) part of the propagating mode of the channel Green
//! function, -i A0 E(z, zeta) [Q_even(|dx|, |y - eta|) + Q_odd(|dx|, |y + eta|)],
//! from the lattice sum of period d = 2W
//!   F(x, s) = sum_j H0^(2)(k sqrt(x^2 + (s - j d)^2))
//! (the 2D Helmholtz Green function of the periodic strip, time factor e^{iwt}):
//!   Q_even(x, u) = F(x, u) - H0^(2)(k |(x, u)|),
//!   Q_odd(x, u)  = F(x, u - W) - H0^(2)(k |(x, u - W)|) - H0^(2)(k |(x, u + W)|).
//! F is evaluated with the Ewald parameter a = min(0.4, 2/(k d)) (Linton's a~, our choice; the bound
//! k a d / 2 <= 1 keeps the two parts of the Ewald form of moderate size, see channel_lattice)
//! by the eigenfunction (strip-mode) series for x >= 2 a d and by the Ewald-type
//! representation of Linton (1999, eqs. 2.26-2.28) below; Q_even and Q_odd are smooth and are
//! tabulated per frequency. Why images, the uniform reflection factor and these truncations:
//! theory §8.5 (the Faddeeva function to 7e-16 with N = 40, F to 1.5e-14, the tiles to 1.6e-11
//! against arbitrary-precision references).

use std::sync::LazyLock;

use num_complex::Complex64;

use crate::constants::{GAMMA, PI};
use crate::error::{Error, Result};
use crate::green_inf::bessel_jy01;
use crate::quadrature::{chebyshev_cosines, chebyshev_nodes, chebyshev_power_basis};

const I: Complex64 = Complex64::new(0.0, 1.0);

// Coefficients of Weideman's rational approximation (SIAM J. Numer. Anal. 31,
// 1497-1518, 1994, the algorithm "cef" with N terms): p(Z) = sum_{j=1..N}
// a_j Z^{j-1}, a_j = (1/2M) sum_{k=-M..M-1} f(k) cos(pi k j / M), M = 2N,
// f(k) = exp(-t^2)(L^2 + t^2) at t = L tan(k pi / 2M), f(-M) = 0,
// L = sqrt(N / sqrt 2).
const WN: usize = 40;

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct WeidemanCoeffs {
    L: f64,
    a: [f64; WN],
}

impl WeidemanCoeffs {
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn new() -> WeidemanCoeffs {
        let M = 2 * WN as i32;
        let L = (WN as f64 / 2.0f64.sqrt()).sqrt();
        let mut f = vec![0.0f64; 2 * M as usize]; // f[k + M]
        for k in -M + 1..=M - 1 {
            let t = L * (k as f64 * PI / (2.0 * M as f64)).tan();
            f[(k + M) as usize] = (-t * t).exp() * (L * L + t * t);
        }
        let mut a = [0.0f64; WN];
        for j in 1..=WN as i32 {
            let mut s = 0.0;
            for k in -M..=M - 1 {
                s += f[(k + M) as usize] * (PI * k as f64 * j as f64 / M as f64).cos();
            }
            a[j as usize - 1] = s / (2.0 * M as f64);
        }
        WeidemanCoeffs { L, a }
    }
}

static WEIDEMAN: LazyLock<WeidemanCoeffs> = LazyLock::new(WeidemanCoeffs::new);

// Hankel function H0^(2)(x) = J0 - i Y0, x > 0.
#[inline]
fn hankel2_0(x: f64) -> Complex64 {
    let (j0, y0, _, _) = bessel_jy01(x);
    Complex64::new(j0, -y0)
}

// e^{-A^2 - B^2} w(i(A + B)) and e^{-A^2 - B^2} w(i(A - B)) without overflow,
// for Re(lambda) >= 0 or lambda = -i kappa, A = lambda a d / 2, B = x / (a d) >= 0:
// the reflection w(z) = 2 exp(-z^2) - w(-z) is used in the lower half plane,
// where exp(-A^2 - B^2 + (A - B)^2) = exp(-lambda x).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn ewald_pair(A: Complex64, B: f64, lambda: Complex64, x: f64) -> Complex64 {
    let E = (-A * A - B * B).exp();
    let z1 = I * (A + B);
    let z2 = I * (A - B);
    let t1 = if z1.im >= 0.0 { E * faddeeva(z1) } else { 2.0 * (lambda * x).exp() - E * faddeeva(-z1) };
    let t2 = if z2.im >= 0.0 { E * faddeeva(z2) } else { 2.0 * (-lambda * x).exp() - E * faddeeva(-z2) };
    t1 + t2
}

/// Faddeeva function w(z) = exp(-z^2) erfc(-i z) (Weideman 1994, N = 40).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn faddeeva(z: Complex64) -> Complex64 {
    if z.im < 0.0 {
        return 2.0 * (-z * z).exp() - faddeeva(-z);
    }
    let c = &*WEIDEMAN;
    let Lmz = c.L - I * z;
    let Z = (c.L + I * z) / Lmz;
    let mut p = Complex64::new(c.a[WN - 1], 0.0);
    for j in (0..WN - 1).rev() {
        p = p * Z + c.a[j];
    }
    2.0 * p / (Lmz * Lmz) + (1.0 / PI.sqrt()) / Lmz
}

/// Exponential integral E_n(x), n >= 1, x > 0.
pub fn expint_en(n: i32, x: f64) -> f64 {
    // E_n(x) for n >= 1, x > 0 (Abramowitz & Stegun 5.1.4); 1.7e-15 of mpmath at 120 digits for
    // n = 1..40, x = 1e-8..200
    let nf = n as f64;
    if x > 1.0 {
        // A&S 5.1.22 in its even form, E_n = e^-x / (x + n - 1 n / (x + n + 2 - 2 (n + 1) /
        // (x + n + 4 - ...))), evaluated from the bottom at depths doubled until two agree
        let fraction = |depth: i32| {
            let mut t = 0.0;
            for k in (1..=depth).rev() {
                t = -(k as f64) * (n - 1 + k) as f64 / (x + nf + 2.0 * k as f64 + t);
            }
            1.0 / (x + nf + t)
        };
        let mut last = fraction(16);
        let mut depth = 32;
        while depth <= 4096 {
            let v = fraction(depth);
            if (v - last).abs() <= 1e-16 * v.abs() {
                return v * (-x).exp();
            }
            last = v;
            depth *= 2;
        }
        return last * (-x).exp();
    }
    // x <= 1: E_1 by its power series (A&S 5.1.11), then E_{k+1} = (e^-x - x E_k) / k (A&S 5.1.14),
    // which damps an error by x / k <= 1
    let mut sum = 0.0;
    let mut power = 1.0;
    for k in 1..60 {
        let kf = k as f64;
        power *= -x / kf; // (-x)^k / k!
        sum += power / kf;
        if (power / kf).abs() < 1e-17 * sum.abs() {
            break;
        }
    }
    let mut e = -GAMMA - x.ln() - sum;
    let ex = (-x).exp();
    for k in 1..n {
        e = (ex - x * e) / k as f64;
    }
    e
}

/// The strip mode n of the lattice of period d at the wave number k: (alpha_n, q_n),
/// alpha_n = 2 pi n / d, q_n = alpha_n^2 - k^2 (q_n > 0 evanescent, q_n < 0 propagating); at a
/// channel resonance q_n = 0 the mode's 1/lambda_n diverges, so within rounding of it (1e-12 k^2)
/// an error.
fn strip_mode(n: i32, k: f64, d: f64) -> Result<(f64, f64)> {
    let al = 2.0 * PI * n as f64 / d;
    let q = al * al - k * k;
    if q.abs() <= 1e-12 * k * k {
        return Err(Error::numerical("the channel width is a multiple of the half wavelength (resonance)"));
    }
    Ok((al, q))
}

/// Lattice sum F(x, s) for k > 0, period d, (x, s) off the lattice points.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn channel_lattice(x: f64, s: f64, k: f64, d: f64) -> Result<Complex64> {
    let x = x.abs();
    // Ewald parameter (Linton 1999 a~): k a d / 2 <= 1 keeps the two parts of
    // moderate size; the eigenfunction series is used for x >= 2 a d
    let a = 0.4f64.min(2.0 / (k * d));
    if x >= 2.0 * a * d {
        // F = (2i/d) sum_n eps_n cos(alpha_n s) e^{-lambda_n x} / lambda_n, lambda_n = sqrt(q_n)
        // (e^{iwt}: i sqrt(-q_n) for the propagating modes), until the first term below
        // e^-40 (lambda_n x > 40), which is reached by n = d sqrt(k^2 + (40/x)^2) / (2 pi) + 1
        let last = (d * (k * k + (40.0 / x) * (40.0 / x)).sqrt() / (2.0 * PI)).ceil() as i32 + 2;
        let mut sum = Complex64::new(0.0, 0.0);
        for n in 0..=last {
            let (_, q) = strip_mode(n, k, d)?;
            let lb = if q > 0.0 { Complex64::new(q.sqrt(), 0.0) } else { Complex64::new(0.0, (-q).sqrt()) };
            let eps = if n == 0 { 1.0 } else { 2.0 };
            let t = eps * (2.0 * PI * n as f64 * s / d).cos() * (-lb * x).exp() / lb;
            sum += t;
            if lb.re > 0.0 && lb.re * x > 40.0 {
                return Ok(Complex64::new(0.0, 2.0 / d) * sum);
            }
        }
        return Err(Error::internal("the strip-mode series of the channel did not reach its stop"));
    }
    // Linton (1999) eqs. 2.26-2.28 in the time factor e^{-iwt} (lambda = -i kappa
    // for the propagating modes), then F = 4 i conj(psi)
    let mut lam_sum = Complex64::new(0.0, 0.0);
    let B = x / (a * d);
    // the spectral terms decay like exp(-(lambda_n a d / 2)^2), lambda_n ~ 2 pi n / d: at
    // n = 3/a about exp(-9 pi^2) = 3e-39 of the first ones
    let nmax = (3.0 / a) as i32 + 4;
    for n in 0..=nmax {
        let (al, q) = strip_mode(n, k, d)?;
        let lam = if q > 0.0 { Complex64::new(q.sqrt(), 0.0) } else { Complex64::new(0.0, -(-q).sqrt()) };
        let A = lam * (a * d / 2.0);
        let L0n = ewald_pair(A, B, lam, x) / (4.0 * lam);
        lam_sum += (if n == 0 { 1.0 } else { 2.0 }) * L0n * (al * s).cos();
    }
    let mut psi = lam_sum / d;
    // sum_n Y~_n, Y~_n = (1/4pi) sum_p (k a d / 2)^{2p} / p! E_{p+1}(R_n^2 / (a d)^2): a term is
    // below e^-X, so the lattice points with X = R_n^2 / (a d)^2 > 45 (e^-45 = 3e-20) are left out;
    // with n0 = floor(s / d) every n outside n0-3..n0+4 has |s - n d| >= 4 d > sqrt(45) a d
    // (a <= 0.4), so the window holds all the others
    let c2 = (k * a * d / 2.0) * (k * a * d / 2.0);
    let n0 = (s / d).floor() as i32;
    for n in n0 - 3..=n0 + 4 {
        let nd = n as f64 * d;
        let R2 = x * x + (s - nd) * (s - nd);
        let X = R2 / (a * a * d * d);
        if X > 45.0 {
            continue;
        }
        let mut term = 1.0;
        let mut y = 0.0;
        // c2 <= 1, so term <= 1/p! and term E_{p+1}(X) <= y / p! (E_n decreases with n): the stop
        // below is reached by p = 19, within the orders n <= 40 at which expint_en is verified
        for p in 0..40 {
            if p > 0 {
                term *= c2 / p as f64;
            }
            let e = expint_en(p + 1, X);
            y += term * e;
            if term * e < 1e-17 * y.abs() {
                break;
            }
        }
        psi += y / (4.0 * PI);
    }
    Ok(4.0 * I * psi.conj())
}

/// Per-frequency tiles of Q_even and Q_odd on [0, X] x [0, W] (Chebyshev,
/// value only; derivatives from the polynomials).
#[derive(Clone, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct ChannelTail {
    pub W: f64,
    pub k: f64,
    pub X: f64,
    pub hx: f64,
    pub hu: f64,
    pub nx: i32,
    pub nu: i32,
    pub even: Vec<Complex64>, // [tile][i][j] power-basis coefficients in (tx, tu)
    pub odd: Vec<Complex64>,
}

const TN: usize = ChannelTail::DEG as usize + 1;

impl ChannelTail {
    pub const DEG: i32 = 11;

    // The tile of (x, u): its power-basis coefficients and the tile variables (tx, tu).
    fn tile(&self, odd_part: bool, x: f64, u: f64) -> (&[Complex64], f64, f64) {
        // the tiles cover [0, X] only; outside they would extrapolate
        assert!(
            x <= self.X * (1.0 + 1e-9) + 1e-12 * self.W,
            "channel tail evaluated beyond its tiles (x = {x:.6} > {:.6})",
            self.X
        );
        let ux = x / self.hx;
        let uu = u / self.hu;
        let ix = (ux as i32).min(self.nx - 1);
        let iu = (uu as i32).min(self.nu - 1);
        let tx = 2.0 * (ux - ix as f64) - 1.0;
        let tu = 2.0 * (uu - iu as f64) - 1.0;
        let off = (ix as usize * self.nu as usize + iu as usize) * TN * TN;
        (&(if odd_part { &self.odd } else { &self.even })[off..off + TN * TN], tx, tu)
    }

    /// q and its derivatives in x and u at (x, u), x in [0, X], u in [0, W]: (q, q_x, q_u).
    pub fn eval(&self, odd_part: bool, x: f64, u: f64) -> (Complex64, Complex64, Complex64) {
        const N: usize = TN;
        let (c, tx, tu) = self.tile(odd_part, x, u);
        let zero = Complex64::new(0.0, 0.0);
        let mut v = zero;
        let mut vx = zero;
        let mut vu = zero;
        for i in (0..N).rev() {
            let r = &c[i * N..i * N + N];
            let mut p = r[N - 1];
            let mut dp = zero;
            for j in (0..N - 1).rev() {
                dp = dp * tu + p;
                p = p * tu + r[j];
            }
            vx = vx * tx + v;
            v = v * tx + p;
            vu = vu * tx + dp;
        }
        (v, vx * (2.0 / self.hx), vu * (2.0 / self.hu))
    }

    /// The second derivatives of q at (x, u) from the same polynomials: (q_xx, q_xu, q_uu).
    pub fn eval2(&self, odd_part: bool, x: f64, u: f64) -> (Complex64, Complex64, Complex64) {
        const N: usize = TN;
        let (c, tx, tu) = self.tile(odd_part, x, u);
        let zero = Complex64::new(0.0, 0.0);
        // Horner in tx of the polynomials in tu, their first and second derivatives
        let (mut v, mut vx, mut vxx, mut vu, mut vxu, mut vuu) = (zero, zero, zero, zero, zero, zero);
        for i in (0..N).rev() {
            let r = &c[i * N..i * N + N];
            let (mut p, mut dp, mut ddp) = (r[N - 1], zero, zero);
            for j in (0..N - 1).rev() {
                ddp = ddp * tu + 2.0 * dp;
                dp = dp * tu + p;
                p = p * tu + r[j];
            }
            vxx = vxx * tx + 2.0 * vx;
            vx = vx * tx + v;
            v = v * tx + p;
            vxu = vxu * tx + vu;
            vu = vu * tx + dp;
            vuu = vuu * tx + ddp;
        }
        let (sx, su) = (2.0 / self.hx, 2.0 / self.hu);
        (vxx * (sx * sx), vxu * (sx * su), vuu * (su * su))
    }
}

/// The tiles of Q_even and Q_odd of the channel of width W at the wave number k, x in [0, X].
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn make_channel_tail(W: f64, k: f64, X: f64) -> Result<ChannelTail> {
    let mut t = ChannelTail { W, k, X: X.max(1e-6 * W), ..Default::default() };
    let lambda = 2.0 * PI / k;
    let h = (lambda / 2.0).min(W);
    t.nx = 1.max((t.X / h).ceil() as i32);
    t.nu = 1.max((W / h).ceil() as i32);
    t.hx = t.X / t.nx as f64;
    t.hu = W / t.nu as f64;
    const N: usize = TN;
    let d = 2.0 * W;
    let nodes = chebyshev_nodes::<N>();
    let a = chebyshev_power_basis::<N>();
    let cs = chebyshev_cosines::<N>();
    let zero = Complex64::new(0.0, 0.0);
    for part in 0..2 {
        let mut out = vec![zero; t.nx as usize * t.nu as usize * N * N];
        for ix in 0..t.nx as usize {
            for iu in 0..t.nu as usize {
                let mut f = [[zero; N]; N];
                for i in 0..N {
                    for j in 0..N {
                        let x = t.hx * (ix as f64 + 0.5 * (nodes[i] + 1.0));
                        let u = t.hu * (iu as f64 + 0.5 * (nodes[j] + 1.0));
                        if part == 0 {
                            f[i][j] = channel_lattice(x, u, k, d)? - hankel2_0(k * x.hypot(u));
                        } else {
                            f[i][j] = channel_lattice(x, u - W, k, d)?
                                - hankel2_0(k * x.hypot(u - W))
                                - hankel2_0(k * x.hypot(u + W));
                        }
                    }
                }
                let mut c = [[zero; N]; N];
                for ji in 0..N {
                    for jj in 0..N {
                        let mut acc = zero;
                        for i in 0..N {
                            for j in 0..N {
                                acc += f[i][j] * (cs[ji][i] * cs[jj][j]);
                            }
                        }
                        c[ji][jj] = acc
                            * ((if ji != 0 { 2.0 } else { 1.0 }) * (if jj != 0 { 2.0 } else { 1.0 }) / (N * N) as f64);
                    }
                }
                let off = (ix * t.nu as usize + iu) * N * N;
                let pw = &mut out[off..off + N * N];
                for ji in 0..N {
                    for jj in 0..N {
                        for ki in 0..=ji {
                            for kj in 0..=jj {
                                pw[ki * N + kj] += c[ji][jj] * (a[ji][ki] * a[jj][kj]);
                            }
                        }
                    }
                }
            }
        }
        if part == 0 {
            t.even = out;
        } else {
            t.odd = out;
        }
    }
    Ok(t)
}
