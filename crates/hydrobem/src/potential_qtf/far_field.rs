//! The fields beyond the panelled zone (theory §12.9): the far-field series fitted on two
//! circles (an outgoing wave and a local term per angular order), the quadrature zone's fields
//! interpolated in 1/r through the circles of the fit, and their values on circles. Why the local
//! term (the propagating series alone moved the sums with R as R^-2) and why the zone from circles
//! with the wave's coefficient from their divided differences (6e-9 where the wave of the outer fit
//! left 4e-3): theory §12.9.

use num_complex::Complex64;

use super::bessel::{bessel_k_scaled, hankel2};
use super::{CZ, I};
use crate::constants::PI;
use crate::dispersion::dispersion_xn;
use crate::env_hooks::validation_constant;

/// The local (non-propagating) part of a disturbance's far field, per angular order m: in finite
/// depth the first evanescent mode K_m(k_1 r) (k_1 tan(k_1 h) = −K), the slowest of the decaying
/// modes; in deep water (and where k_1 R < 2, a depth large against the radius) the leading term
/// of the local field's expansion in 1/r, r^{−3−|m|} (the non-oscillating part 2/(K²R³) of the
/// Green function at the free surface and its expansion in the source positions).
#[derive(Clone, Copy)]
pub(super) enum Local {
    Power,
    Evanescent(f64),
}

/// The orders that carry a local term (the higher ones are below the rounding of the fit).
pub(super) const LOCAL_ORDERS: usize = 30;

/// The radius of the fit's second circle over its first (1.1; a validation hook can vary it).
pub(super) fn fit_ratio() -> f64 {
    validation_constant("FIT_RADIUS_RATIO")
}

impl Local {
    #[allow(non_snake_case)] // the notation of the theory documents
    pub(super) fn of(kk: f64, depth: f64, R: f64) -> Local {
        if depth.is_finite() {
            let k1 = dispersion_xn(kk * depth, 1) / depth;
            if k1 * R >= 2.0 {
                return Local::Evanescent(k1);
            }
        }
        Local::Power
    }

    /// [L, dL/dr, e^{−imθ} ∇_h²(L e^{imθ})] of the orders m = 0..LOCAL_ORDERS at r, L(R) = 1.
    #[allow(non_snake_case)] // the notation of the theory documents
    pub(super) fn basis(&self, r: f64, R: f64) -> Vec<[f64; 3]> {
        let n = LOCAL_ORDERS;
        match *self {
            Local::Power => (0..=n)
                .map(|m| {
                    let p = 3.0 + m as f64;
                    let l = (R / r).powf(p);
                    [l, -p / r * l, (p * p - (m * m) as f64) / (r * r) * l]
                })
                .collect(),
            Local::Evanescent(k1) => {
                let kr = bessel_k_scaled(n + 1, k1 * r);
                let kR = bessel_k_scaled(n + 1, k1 * R);
                let e = (-k1 * (r - R)).exp();
                (0..=n)
                    .map(|m| {
                        let l = kr[m] / kR[m] * e;
                        let km1 = if m == 0 { kr[1] } else { kr[m - 1] };
                        let dk = -0.5 * (km1 + kr[m + 1]) / kr[m]; // K_m'/K_m
                        [l, k1 * dk * l, k1 * k1 * l]
                    })
                    .collect()
            }
        }
    }
}

/// The radial functions of a far field at r: H2_m(kr), k H2_m'(kr) (index m + M) and the local
/// basis.
pub(super) type Radial = (Vec<Complex64>, Vec<Complex64>, Vec<[f64; 3]>);

/// The radial functions at r of an outgoing field of wavenumber k and local term `local` (the
/// local basis normalized at R) to the order mm.
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn outgoing_radial(k: f64, local: Local, r: f64, R: f64, mm: usize) -> Radial {
    let (h, hd) = hankel2(mm, k * r);
    let hd = hd.into_iter().map(|x| x * k).collect();
    (h, hd, local.basis(r, R))
}

/// The linear map of a far-field fit (theory §12.9) per order m (index m + M): from the
/// discrete Fourier coefficients c1, c2 of the circles R and R2 to the coefficients of the wave and
/// the local term, a_m = A1 c1 + A2 c2, b_m = B1 c1 + B2 c2, [A1, A2, B1, B2] (the 2 × 2 system of
/// each order; b_m = 0 beyond LOCAL_ORDERS and where the two radial functions are nearly dependent
/// between the circles; all zero for an order whose Hankel function overflows at the circles, a
/// high order at a small wavenumber, which gets no field).
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn fit_map(mm: usize, k: f64, kk: f64, R: f64, R2: f64, depth: f64) -> (Local, Vec<[Complex64; 4]>) {
    let local = Local::of(kk, depth, R);
    let (h1, _) = hankel2(mm, k * R);
    let (h2, _) = hankel2(mm, k * R2);
    let l2 = local.basis(R2, R);
    let one = Complex64::new(1.0, 0.0);
    // the validation hooks of the sensitivity studies (the method's 30 and 0.05 without them)
    let orders = (validation_constant("FIT_LOCAL_ORDERS").max(-1.0) as i64).min(LOCAL_ORDERS as i64);
    let dependence = validation_constant("FIT_DEPENDENCE");
    let map = (-(mm as i64)..=(mm as i64))
        .enumerate()
        .map(|(i, m)| {
            if !(h1[i].is_finite() && h2[i].is_finite()) {
                return [CZ; 4];
            }
            let ma = m.unsigned_abs() as usize;
            // the local term only for |m| <= LOCAL_ORDERS and where the two radial functions are
            // not nearly dependent between the circles (the 0.05 below; theory §12.9, the fit)
            let local_order = (ma as i64) <= orders;
            let det = if local_order { h1[i] * l2[ma][0] - h2[i] } else { CZ };
            if local_order && det.norm() > dependence * ((h1[i] * l2[ma][0]).norm() + h2[i].norm()) {
                [cdiv(l2[ma][0] * one, det), cdiv(-one, det), cdiv(-h2[i], det), cdiv(h1[i], det)]
            } else {
                [cdiv(one, h1[i]), CZ, CZ, CZ]
            }
        })
        .collect();
    (local, map)
}

/// A field's far field beyond R: Σ_m [a_m H2_m(kr) + b_m L_m(r)] e^{imθ} (index m + M), fitted on
/// the circles R and R2 (fit_map).
pub(super) struct Far {
    a: Vec<Complex64>,
    b: Vec<Complex64>,
    k: f64,
    pub(super) kk: f64,
    local: Local,
}

impl Far {
    /// c1, c2: the discrete Fourier coefficients (ring_dft) of the circles R and R2 to the order mm;
    /// k the wavenumber, kk = ω²/g, depth the water depth.
    #[allow(non_snake_case)] // the notation of the theory documents
    #[allow(clippy::too_many_arguments)] // the operands of the fit, passed as they are
    fn fit(c1: &[Complex64], c2: &[Complex64], mm: usize, k: f64, kk: f64, R: f64, R2: f64, depth: f64) -> Far {
        let (local, map) = fit_map(mm, k, kk, R, R2, depth);
        let a = map.iter().zip(c1.iter().zip(c2)).map(|(p, (x, y))| p[0] * x + p[1] * y).collect();
        let b = map.iter().zip(c1.iter().zip(c2)).map(|(p, (x, y))| p[2] * x + p[3] * y).collect();
        Far { a, b, k, kk, local }
    }

    /// The series to the order mm ≤ its own (the asymptotic zone's truncation).
    pub(super) fn truncated(&self, mm: usize) -> Far {
        let m0 = (self.a.len() - 1) / 2;
        let r = m0 - mm..=m0 + mm;
        Far { a: self.a[r.clone()].to_vec(), b: self.b[r].to_vec(), k: self.k, kk: self.kk, local: self.local }
    }

    #[allow(non_snake_case)] // the notation of the theory documents
    pub(super) fn radial(&self, r: f64, R: f64, mm: usize) -> Radial {
        outgoing_radial(self.k, self.local, r, R, mm)
    }

    /// Per order (index m + M) the coefficients of e^{imθ} at the radius of rad: the potential,
    /// its r derivative and γ = −g∇_h²φ − ω²∂_zφ (∂_zφ = Kφ on z = 0 for both parts).
    pub(super) fn modes(&self, rad: &Radial, g: f64) -> Vec<[Complex64; 3]> {
        let (h, hd, l) = rad;
        let mm = (self.a.len() - 1) / 2;
        (-(mm as i64)..=(mm as i64))
            .enumerate()
            .map(|(i, m)| {
                let ma = m.unsigned_abs() as usize;
                let [lv, ld, ll] = if ma <= LOCAL_ORDERS { l[ma] } else { [0.0; 3] };
                // an order without a wave (its Hankel function overflowed at the fit) adds none
                let (wh, wd) = if self.a[i] == CZ { (CZ, CZ) } else { (self.a[i] * h[i], self.a[i] * hd[i]) };
                let p = wh + self.b[i] * lv;
                let pr = wd + self.b[i] * ld;
                let gm = g * (self.k * self.k - self.kk * self.kk) * wh - g * (ll + self.kk * self.kk * lv) * self.b[i];
                [p, pr, gm]
            })
            .collect()
    }
}

/// The discrete Fourier coefficients (1/n) Σ_t v_t e^{−imθ_t} of the values v on the n equally
/// spaced points θ_t = 2πt/n of a circle, m = −mm..mm (index m + mm), the phase by its recurrence.
pub(super) fn ring_dft(v: &[Complex64], mm: usize) -> Vec<Complex64> {
    let n = v.len();
    (-(mm as i64)..=(mm as i64))
        .map(|m| {
            let w = Complex64::from_polar(1.0, -(m as f64) * 2.0 * PI / n as f64);
            let (mut c, mut e) = (CZ, Complex64::new(1.0, 0.0));
            for x in v {
                c += x * e;
                e *= w;
            }
            c / n as f64
        })
        .collect()
}

/// The weights on the n points θ_t = 2πt/n of a circle of a functional Σ_m μ_m c_m of its discrete
/// Fourier coefficients c_m (ring_dft, index m + mm): (1/n) Σ_m μ_m e^{−imθ_t}.
pub(super) fn ring_weights(mu: &[Complex64], n: usize) -> Vec<Complex64> {
    let mm = (mu.len() - 1) / 2;
    let mut w = vec![CZ; n];
    for (i, m) in (-(mm as i64)..=(mm as i64)).enumerate() {
        if mu[i] == CZ {
            continue;
        }
        let step = Complex64::from_polar(1.0, -(m as f64) * 2.0 * PI / n as f64);
        let mut e = mu[i] / n as f64;
        for x in w.iter_mut() {
            *x += e;
            e *= step;
        }
    }
    w
}

/// a / b scaled by |b| first: the Hankel functions of high orders at small arguments exceed
/// 1e154, where |b|² of the plain complex division overflows (and a moment of the same size gives
/// inf / inf).
pub(super) fn cdiv(a: Complex64, b: Complex64) -> Complex64 {
    let s = b.norm();
    (a / s) / (b / s)
}

/// The circles of the fit beyond the panelled zone (theory §12.9): ρ_0 = R < ... < ρ_{n−1} = R_OC
/// at the Chebyshev-Lobatto nodes in u = 1/r (without a quadrature zone R_OC = R, n = 1), the
/// polynomial interpolation through them in u and the weights of their divided difference.
pub(super) struct Circles {
    pub(super) rho: Vec<f64>,
    u: Vec<f64>,             // 1/ρ_i
    pub(super) dd: Vec<f64>, // f[u_0, ..., u_{n−1}] = Σ_i dd_i f(u_i), dd_i = 1/Π_{j≠i}(u_i − u_j)
}

impl Circles {
    #[allow(non_snake_case)] // the notation of the theory documents
    pub(super) fn new(R: f64, Ro: f64, n: usize) -> Circles {
        let rho: Vec<f64> = if n < 2 {
            vec![R]
        } else {
            let (u0, u1) = (1.0 / Ro, 1.0 / R);
            (0..n)
                .map(|i| {
                    if i == 0 {
                        R
                    } else if i == n - 1 {
                        Ro
                    } else {
                        1.0 / (0.5 * (u0 + u1) + 0.5 * (u1 - u0) * (PI * i as f64 / (n - 1) as f64).cos())
                    }
                })
                .collect()
        };
        let u: Vec<f64> = rho.iter().map(|r| 1.0 / r).collect();
        let dd = (0..u.len())
            .map(|i| 1.0 / (0..u.len()).filter(|&j| j != i).map(|j| u[i] - u[j]).product::<f64>())
            .collect();
        Circles { rho, u, dd }
    }

    /// The Lagrange polynomials in u = 1/r of the circles at r and their derivatives along r:
    /// (ℓ_i(1/r), dℓ_i(1/r)/dr).
    fn lagrange(&self, r: f64) -> (Vec<f64>, Vec<f64>) {
        let (u, n) = (1.0 / r, self.u.len());
        let mut l = vec![1.0; n];
        let mut d = vec![0.0; n];
        for i in 0..n {
            for j in (0..n).filter(|&j| j != i) {
                l[i] *= (u - self.u[j]) / (self.u[i] - self.u[j]);
                // the product rule: the factor j differentiated, the others kept
                let mut p = 1.0 / (self.u[i] - self.u[j]);
                for k in (0..n).filter(|&k| k != i && k != j) {
                    p *= (u - self.u[k]) / (self.u[i] - self.u[k]);
                }
                d[i] += p;
            }
            d[i] *= -u * u; // d/dr = −u² d/du
        }
        (l, d)
    }

    /// The divided difference of H2_m(kr) over the circles per order (index m + M): the
    /// denominator of the wave coefficient (Zone), zero where the Hankel function overflows on a
    /// circle (a high order at a small wavenumber, which gets no wave).
    pub(super) fn wave_denominators(&self, k: f64, mm: usize) -> Vec<Complex64> {
        let hr: Vec<Vec<Complex64>> = self.rho.iter().map(|&r| hankel2(mm, k * r).0).collect();
        (0..2 * mm + 1)
            .map(|i| {
                let d: Complex64 = hr.iter().zip(&self.dd).map(|(h, w)| h[i] * w).sum();
                if d.is_finite() { d } else { CZ }
            })
            .collect()
    }
}

/// A field beyond the panelled zone (theory §12.9): its discrete Fourier coefficients c_{i,m}
/// on the circles ρ_i (Circles) and its far-field series (Far) fitted on the outer circle ρ_{n−1}
/// and 1.1 ρ_{n−1}. Between R and R_OC per order the outgoing wave a_m H2_m(kr) plus the remainder,
/// a polynomial in 1/r: a_m from the divided differences over the circles, a_m = c[u_0..u_{n−1}] /
/// H_m[u_0..u_{n−1}], so that c − a H interpolated through the circles is of degree n − 2; beyond
/// R_OC the series.
pub(super) struct Zone {
    pub(super) far: Far,
    c: Vec<Vec<Complex64>>, // [i][m + M] on the circles of the zone
    a: Vec<Complex64>,      // the wave coefficient of the zone per order
}

/// The radial functions of a zone's field of the wavenumber k at r per order (index m + M):
/// H2_m(kr) − Σ_i ℓ_i H2_m(kρ_i) and its r derivative (the wave less its interpolant), the Lagrange
/// polynomials ℓ_i(1/r) and their r derivatives (Circles::lagrange).
pub(super) type ZoneRadial = (Vec<Complex64>, Vec<Complex64>, Vec<f64>, Vec<f64>);

impl Zone {
    /// rings[i]: the discrete Fourier coefficients of the circles ρ_0..ρ_{n−1} and last of 1.1 ρ_{n−1};
    /// den the wave denominators of k (Circles::wave_denominators).
    #[allow(clippy::too_many_arguments)] // the operands of the fit, passed as they are
    pub(super) fn fit(
        mut rings: Vec<Vec<Complex64>>,
        ci: &Circles,
        den: &[Complex64],
        mm: usize,
        k: f64,
        kk: f64,
        depth: f64,
    ) -> Zone {
        let n = ci.rho.len();
        let ro = ci.rho[n - 1];
        let far = Far::fit(&rings[n - 1], &rings[n], mm, k, kk, ro, fit_ratio() * ro, depth);
        rings.truncate(n);
        let a = (0..2 * mm + 1)
            .map(|i| {
                if n < 2 || den[i] == CZ {
                    return CZ;
                }
                let s: Complex64 = rings.iter().zip(&ci.dd).map(|(c, w)| c[i] * w).sum();
                cdiv(s, den[i])
            })
            .collect();
        Zone { far, c: rings, a }
    }

    /// The radial functions at r of the wavenumber k (ZoneRadial).
    pub(super) fn radial(ci: &Circles, k: f64, r: f64, mm: usize) -> ZoneRadial {
        let (h, hd) = hankel2(mm, k * r);
        let (l, d) = ci.lagrange(r);
        let hr: Vec<Vec<Complex64>> = ci.rho.iter().map(|&x| hankel2(mm, k * x).0).collect();
        let (mut g0, mut g1) = (h, hd.into_iter().map(|x| x * k).collect::<Vec<_>>());
        for (j, hj) in hr.iter().enumerate() {
            for i in 0..g0.len() {
                g0[i] -= l[j] * hj[i];
                g1[i] -= d[j] * hj[i];
            }
        }
        (g0, g1, l, d)
    }

    /// Per order (index m + M) the potential and its r derivative at the radius of rad (R ≤ r ≤
    /// R_OC; the third entry, γ, unused there and zero).
    pub(super) fn modes(&self, rad: &ZoneRadial) -> Vec<[Complex64; 3]> {
        let (g0, g1, l, d) = rad;
        (0..self.a.len())
            .map(|i| {
                let (mut p, mut pr) = (CZ, CZ);
                for (j, c) in self.c.iter().enumerate() {
                    p += l[j] * c[i];
                    pr += d[j] * c[i];
                }
                if self.a[i] != CZ {
                    p += self.a[i] * g0[i];
                    pr += self.a[i] * g1[i];
                }
                [p, pr, CZ]
            })
            .collect()
    }
}

/// φ and ∇_h φ of a field on the points θ_t of the circle r from its modes md at r (Far::modes,
/// Zone::modes; index m + M): e the table e^{imθ_t} [t][m + M], cs (cos θ_t, sin θ_t).
pub(super) fn on_circle(
    md: &[[Complex64; 3]],
    r: f64,
    e: &[Complex64],
    cs: &[(f64, f64)],
) -> Vec<(Complex64, [Complex64; 2])> {
    let nm = md.len();
    let mm = (nm - 1) / 2;
    cs.iter()
        .enumerate()
        .map(|(t, &(c, s))| {
            let (mut p, mut pr, mut pt) = (CZ, CZ, CZ);
            for (i, (x, y)) in md.iter().zip(&e[t * nm..(t + 1) * nm]).enumerate() {
                let v = x[0] * y;
                p += v;
                pr += x[1] * y;
                pt += (i as f64 - mm as f64) * v;
            }
            let pt = I * pt / r;
            (p, [c * pr - s * pt, s * pr + c * pt])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quadrature_zone_from_its_circles() {
        // a field of the zone's kind per order: an outgoing wave, a multipole and an evanescent
        // mode, f_m(r) = a H_m(kr) + b (R/r)^(1+|m|) + d K_m(2r)/K_m(2R) (m = -6..6, k R = 4.5, so
        // that the orders above k R grow into the inner part of the zone), given on the 13 circles
        // of 3 <= r <= 9 only: Zone::modes reproduces it and its r derivative between them, of each
        // order's largest value. Without the evanescent mode exactly (the divided difference finds
        // a, the remainder is a polynomial in 1/r of degree 7 <= n - 2): 1e-10; with it 2.4e-7 (its
        // interpolation in 1/r)
        let (k, r_in, r_oc, mm): (f64, f64, f64, usize) = (1.5, 3.0, 9.0, 6);
        let n = 6 + (4.0 * (r_oc / r_in).log2()).ceil() as usize;
        let ci = Circles::new(r_in, r_oc, n);
        let coef = |i: usize| {
            let m = i as f64 - mm as f64;
            (Complex64::new(0.3 + 0.1 * m, -0.2), Complex64::new(1.0, 0.5 * m), Complex64::new(-0.4, 0.2))
        };
        for (ev, tol) in [(0.0, 1e-10), (1.0, 1e-6)] {
            let field = |r: f64| -> (Vec<Complex64>, Vec<Complex64>) {
                let (h, hd) = hankel2(mm, k * r);
                let (kv, kv0) = (bessel_k_scaled(mm + 1, 2.0 * r), bessel_k_scaled(mm + 1, 2.0 * r_in));
                let e = (-2.0 * (r - r_in)).exp();
                (0..2 * mm + 1)
                    .map(|i| {
                        let ma = (i as i64 - mm as i64).unsigned_abs() as usize;
                        let (a, b, d) = coef(i);
                        let p = 1.0 + ma as f64;
                        let l = kv[ma] / kv0[ma] * e;
                        let km1 = if ma == 0 { kv[1] } else { kv[ma - 1] };
                        let dl = -(km1 + kv[ma + 1]) / kv[ma] * l; // 2 K_m'/K_m, K_m' = -(K_{m-1} + K_{m+1})/2
                        let v = a * h[i] + b * (r_in / r).powf(p) + ev * d * l;
                        let dv = a * k * hd[i] - b * p / r * (r_in / r).powf(p) + ev * d * dl;
                        (v, dv)
                    })
                    .unzip()
            };
            let mut rings: Vec<Vec<Complex64>> = ci.rho.iter().map(|&r| field(r).0).collect();
            rings.push(field(1.1 * r_oc).0);
            let den = ci.wave_denominators(k, mm);
            let z = Zone::fit(rings, &ci, &den, mm, k, k, f64::INFINITY);
            // the errors of each order against its largest value in the zone
            let (mut err, mut big) = (vec![[0.0f64; 2]; 2 * mm + 1], vec![[0.0f64; 2]; 2 * mm + 1]);
            for t in 0..40 {
                let r = r_in + (r_oc - r_in) * (t as f64 + 0.5) / 40.0;
                let md = z.modes(&Zone::radial(&ci, k, r, mm));
                let (v, dv) = field(r);
                for i in 0..2 * mm + 1 {
                    err[i] = [err[i][0].max((md[i][0] - v[i]).norm()), err[i][1].max((md[i][1] - dv[i]).norm())];
                    big[i] = [big[i][0].max(v[i].norm()), big[i][1].max(dv[i].norm())];
                }
            }
            let worst = err.iter().zip(&big).map(|(e, b)| (e[0] / b[0]).max(e[1] / b[1])).fold(0.0, f64::max);
            assert!(worst < tol, "{ev} {worst}");
        }
    }
}
