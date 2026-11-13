//! Roots of the dispersion relation omega^2 = g k tanh(k h) and of its evanescent modes.
//!
//! Finite-depth dispersion relation (theory §1.1, theory §8), nondimensional H = K h,
//! K = w^2 / g:
//!   propagating mode   x0 tanh x0 = H,            x0 = k0 h > 0;
//!   evanescent modes   xn tan xn = -H,            (n - 1/2) pi < xn < n pi.
//! Both are solved to full double precision (test-plan A3).

use crate::constants::PI;

/// Propagating root x0 = k0 h of x0 tanh x0 = H, for H = K h >= 0 (0 for H = 0, H for large H).
///
/// f(x) = x tanh x - H is increasing on x > 0 with
/// max(H, sqrt H) <= x0 <= (H + sqrt(H^2 + 4H)) / 2 (from x/(1+x) < tanh x < min(1, x)).
/// Newton safeguarded by bisection on this bracket; the iteration stops when
/// the step no longer changes x or returns to the previous iterate.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn dispersion_x0(H: f64) -> f64 {
    if !(H > 0.0) {
        return 0.0;
    }
    if H >= 40.0 {
        // tanh x0 = 1 - 2 exp(-2 x0) + ... rounds to 1 for x0 >= 19
        return H;
    }
    let mut lo = H.max(H.sqrt());
    let mut hi = 0.5 * (H + (H * H + 4.0 * H).sqrt());
    let mut x = if H < 1.0 { H.sqrt() * (1.0 + H / 6.0) } else { H }; // x tanh x ~ x^2 (1 - x^2/3)
    x = x.max(lo).min(hi);
    let mut prev = -1.0;
    for _ in 0..200 {
        let t = x.tanh();
        let f = x * t - H;
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let fp = t + x * (1.0 - t * t);
        let mut xn = x - f / fp;
        if !(xn >= lo && xn <= hi) {
            xn = 0.5 * (lo + hi);
        }
        if xn == x || xn == prev {
            break;
        }
        prev = x;
        x = xn;
    }
    x
}

/// Evanescent root xn of xn tan xn = -H, (n - 1/2) pi < xn < n pi, for H = K h >= 0 or +inf
/// and n >= 1 (n pi for H = 0, (n - 1/2) pi for H = inf).
///
/// With xn = n pi - d, d in (0, pi/2): g(d) = (n pi - d) sin d - H cos d = 0,
/// g(0) = -H < 0, g(pi/2) = (n - 1/2) pi > 0, one root: g = 0 is (n pi - d) tan d = H with the left
/// side increasing (g itself need not be, for H < 1). Safeguarded Newton.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn dispersion_xn(H: f64, n: i32) -> f64 {
    let npi = n as f64 * PI;
    if !(H > 0.0) {
        return npi;
    }
    if H.is_infinite() {
        return npi - 0.5 * PI;
    }
    let mut lo = 0.0;
    let mut hi = 0.5 * PI;
    let mut d = (H / (npi - 0.25 * PI)).atan(); // tan d ~ H / (n pi - d)
    let mut prev = -1.0;
    for _ in 0..200 {
        let s = d.sin();
        let c = d.cos();
        let g = (npi - d) * s - H * c;
        if g < 0.0 {
            lo = d;
        } else {
            hi = d;
        }
        let gp = (npi - d) * c + (H - 1.0) * s;
        let mut dn = d - g / gp;
        if !(dn > lo && dn < hi) {
            dn = 0.5 * (lo + hi);
        }
        if dn == d || dn == prev {
            break;
        }
        prev = d;
        d = dn;
    }
    npi - d
}
