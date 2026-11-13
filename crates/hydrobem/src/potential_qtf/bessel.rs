//! The Bessel functions of integer order of the far field (theory §12.9): J_n and Y_n, the
//! Hankel functions H2_m with their derivatives, and the scaled modified functions e^x K_n.

use num_complex::Complex64;

use crate::green_inf::bessel_jy01;

/// J_0..J_n and Y_0..Y_n at x > 0: Y by forward recurrence, J by Miller's backward recurrence
/// normalized by J_0 + 2 Σ J_2k = 1.
pub(super) fn bessel_jy(n: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let (j0, y0, j1, y1) = bessel_jy01(x);
    let mut y = vec![0.0; n + 1];
    y[0] = y0;
    if n >= 1 {
        y[1] = y1;
    }
    for k in 1..n {
        y[k + 1] = 2.0 * k as f64 / x * y[k] - y[k - 1];
    }
    let mut j = vec![0.0; n + 1];
    // the start well above both n and x
    let top = (n.max(x as usize) + 30 + (x.sqrt() * 10.0) as usize) | 1;
    let (mut jp, mut jc) = (0.0f64, 1e-300f64);
    let mut sum = 0.0;
    for k in (1..=top).rev() {
        let jm = 2.0 * k as f64 / x * jc - jp;
        jp = jc;
        jc = jm;
        if k - 1 <= n {
            j[k - 1] = jc;
        }
        if (k - 1) % 2 == 0 && k > 1 {
            sum += 2.0 * jc;
        }
        if jc.abs() > 1e250 {
            // rescale
            jp *= 1e-250;
            jc *= 1e-250;
            sum *= 1e-250;
            for v in j.iter_mut() {
                *v *= 1e-250;
            }
        }
    }
    sum += jc; // J_0
    let s = 1.0 / sum;
    for v in j.iter_mut() {
        *v *= s;
    }
    // the first two from the direct evaluation (more accurate at small x)
    j[0] = j0;
    if n >= 1 {
        j[1] = j1;
    }
    (j, y)
}

/// H2_m(x) = J_m − i Y_m and its derivative for m = −M..M (index m + M), the derivative by
/// H_m' = H_{m−1} − (m/x) H_m (m ≥ 0; H_m' = (m/x) H_m − H_{m+1} for m < 0), which needs no order
/// beyond |m| (the highest orders overflow first at small x).
pub(super) fn hankel2(mm: usize, x: f64) -> (Vec<Complex64>, Vec<Complex64>) {
    let (j, y) = bessel_jy(mm + 1, x);
    let h = |m: i64| -> Complex64 {
        let a = m.unsigned_abs() as usize;
        let s = if m < 0 && a % 2 == 1 { -1.0 } else { 1.0 };
        Complex64::new(s * j[a], -s * y[a])
    };
    let mut v = Vec::with_capacity(2 * mm + 1);
    let mut d = Vec::with_capacity(2 * mm + 1);
    for m in -(mm as i64)..=(mm as i64) {
        v.push(h(m));
        d.push(if m >= 0 { h(m - 1) - (m as f64 / x) * h(m) } else { (m as f64 / x) * h(m) - h(m + 1) });
    }
    (v, d)
}

/// e^x K_0(x) .. e^x K_n(x) at x > 0: K_0 and K_1 by the trapezoidal rule of K_ν(x) =
/// ∫_0^∞ e^{−x cosh t} cosh(νt) dt (an even analytic integrand, so the error falls exponentially
/// with 1/step), the higher orders by the forward recurrence K_{m+1} = K_{m−1} + (2m/x) K_m,
/// stable for the growing K_m.
pub(super) fn bessel_k_scaled(n: usize, x: f64) -> Vec<f64> {
    let step = 0.05f64;
    let (mut k0, mut k1) = (0.5, 0.5); // the half weight of t = 0
    let mut t = step;
    loop {
        let e = (-x * (t.cosh() - 1.0)).exp();
        k0 += e;
        k1 += e * t.cosh();
        if e * t.cosh() < 1e-18 {
            break;
        }
        t += step;
    }
    let mut k = vec![k0 * step, k1 * step];
    for m in 1..n {
        let next = k[m - 1] + 2.0 * m as f64 / x * k[m];
        k.push(next);
    }
    k.truncate(n + 1);
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::PI;

    #[test]
    #[allow(clippy::excessive_precision)] // the reference values as printed, 17 digits
    fn bessel_recurrences() {
        // J_n and Y_n against mpmath.besselj and mpmath.bessely (mpmath 1.4.1, 30 digits, rounded
        // to 17), also for orders far above x, where Y_n grows factorially and J_n falls so: the
        // backward recurrence of J and the forward recurrence of Y to 1e-12 of each value; the
        // Wronskian J_{n+1} Y_n − J_n Y_{n+1} = 2/(π x) to 1e-12 of the size of its products for
        // n < 160 (a wrong recurrence above x fails it there), and J_0 + 2 Σ J_2k = 1 over 240 orders
        let refs: [(f64, usize, f64, f64); 13] = [
            (3.7, 0, -0.39923020337119112, 0.10607431532035411),
            (3.7, 1, 0.053833987745461791, 0.41667437268380749),
            (3.7, 2, 0.42832965620657587, 0.11915507531954182),
            (3.7, 3, 0.40922510004543101, -0.28785807504105958),
            (3.7, 20, 7.6960098267430457e-14, -210444420906.07178),
            (0.3, 5, 6.3044326337710711e-7, -101169.65735231197),
            (0.3, 30, 7.223746206919442e-58, -1.4688861405326529e+55),
            (25.0, 10, -0.075179843948523284, -0.1487183904998065),
            (25.0, 40, 1.674577415562266e-6, -6091.2102591779882),
            (140.0, 60, 0.0062512280177763699, 0.07066640680643145),
            (140.0, 150, 0.0043110999456728775, -1.3902190109679341),
            (0.3, 0, 0.97762624653829609, -0.80727357780451949),
            (140.0, 0, 0.037358225012042691, 0.056139274258306646),
        ];
        for (x, n, jw, yw) in refs {
            let (j, y) = bessel_jy(160, x);
            assert!((j[n] - jw).abs() <= 1e-12 * jw.abs(), "J {x} {n}: {} {jw}", j[n]);
            assert!((y[n] - yw).abs() <= 1e-12 * yw.abs(), "Y {x} {n}: {} {yw}", y[n]);
        }
        for &x in &[0.3, 3.7, 25.0, 140.0] {
            let (j, y) = bessel_jy(240, x);
            for n in 0..160 {
                // (at x = 0.3 Y_n nears the largest double beyond n = 115, where J_n underflows:
                // the test stops there)
                if y[n + 1].abs() >= 1e290 {
                    assert!(x < 1.0 && n > 100, "{x} {n}");
                    break;
                }
                let (p, q) = (j[n + 1] * y[n], j[n] * y[n + 1]);
                assert!((p - q - 2.0 / (PI * x)).abs() <= 1e-12 * (p.abs() + q.abs()), "{x} {n}");
            }
            let s: f64 = j[0] + 2.0 * (1..120).map(|k| j[2 * k]).sum::<f64>();
            assert!((s - 1.0).abs() < 1e-12, "{x} {s}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // the reference values as printed, 17 digits
    fn modified_bessel_k() {
        // e^x K_m(x) against mpmath.besselk(m, x) e^x (mpmath 1.4.1, 30 digits, rounded to 17;
        // orders 0, 1, 5, 30)
        let refs = [
            (0.2, [2.1407573233800412, 5.8333860371867255, 1462025.1983783109, 5.397814618200845e60]),
            (2.0, [0.84156821507077142, 1.0334768470686886, 69.686550876076751, 3.1559587808452645e31]),
            (15.0, [0.32100235350577624, 0.33153489496662908, 0.71520416495590612, 12371483043.881521]),
        ];
        for (x, v) in refs {
            let k = bessel_k_scaled(30, x);
            for (m, want) in [0usize, 1, 5, 30].into_iter().zip(v) {
                assert!((k[m] - want).abs() <= 1e-12 * want, "{x} {m}: {} {want}", k[m]);
            }
        }
    }
}
