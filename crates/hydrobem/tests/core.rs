//! The accuracy tests of the engine components (test-plan A1, A2, A3).
//!
//! `cargo test --release --test core -- --nocapture` prints the error summaries; with the
//! environment variable HB_TIMING set, also the informational kernel timings (not part of a pass
//! or fail). The random points come from the 64-bit Mersenne Twister MT19937-64 (Matsumoto and
//! Nishimura) with a fixed seed and a uniform draw of one 64-bit output (below), so that the
//! points and the error summaries are reproducible.

mod document;
mod ho_b;
mod prep;
mod problem;

use num_complex::Complex64;
use std::fs;
use std::time::Instant;

use hydrobem_core::internals::dispersion::{dispersion_x0, dispersion_xn};
use hydrobem_core::internals::green_channel::{ChannelTail, channel_lattice, faddeeva, make_channel_tail};
use hydrobem_core::internals::green_fin::{
    FreqMode, fin_point, fin4_applies, finite_kernel, finite_kernel_hess, finite_kernel4, make_finite_depth,
    verify_fin_tables,
};
use hydrobem_core::internals::green_inf::{
    bessel_j01, bessel_j01_4, bessel_jy01, bessel_jy01_4, direct4_applies, near4_applies, verify_green_tables,
    wave_kernel, wave_kernel_direct4, wave_kernel_near4, wave_real, wave_real_direct4, wave_x_over_x,
};

mod ho_a;
mod io;
mod kernels;
mod lo_a;
mod lo_b;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");
const PI_REF: f64 = std::f64::consts::PI;

fn read_data(name: &str) -> String {
    fs::read_to_string(format!("{DATA}/{name}")).unwrap_or_else(|e| panic!("{name} is readable: {e}"))
}

/// The data lines of a reference file (no comments, no blank lines), split at white space.
fn data_lines(text: &str) -> impl Iterator<Item = (&str, Vec<&str>)> {
    text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(|l| (l, l.split_whitespace().collect::<Vec<_>>()))
}

fn num(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| panic!("not a number: {s}"))
}

/// MT19937-64, the 64-bit Mersenne Twister (the random points of the tests).
struct Mt64 {
    mt: [u64; 312],
    i: usize,
}

impl Mt64 {
    fn new(seed: u64) -> Mt64 {
        let mut mt = [0u64; 312];
        mt[0] = seed;
        for i in 1..312 {
            mt[i] = 6364136223846793005u64.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62)).wrapping_add(i as u64);
        }
        Mt64 { mt, i: 312 }
    }
    fn next(&mut self) -> u64 {
        const UPPER: u64 = !0u64 << 31;
        const LOWER: u64 = (1u64 << 31) - 1;
        if self.i >= 312 {
            for k in 0..312 {
                let x = (self.mt[k] & UPPER) | (self.mt[(k + 1) % 312] & LOWER);
                let mut xa = x >> 1;
                if x & 1 != 0 {
                    xa ^= 0xB5026F5AA96619E9;
                }
                self.mt[k] = self.mt[(k + 156) % 312] ^ xa;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= (y >> 29) & 0x5555555555555555;
        y ^= (y << 17) & 0x71D67FFFEDA60000;
        y ^= (y << 37) & 0xFFF7EEE000000000;
        y ^= y >> 43;
        y
    }
    /// A uniform value in [a, b): one 64-bit draw divided by 2^64, scaled.
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        let mut u = self.next() as f64 / 18446744073709551616.0;
        if u >= 1.0 {
            u = 1.0 - f64::EPSILON / 2.0;
        }
        u * (b - a) + a
    }
}

/// J_n(x) by the trapezoidal rule of (1/pi) int_0^pi cos(n t - x sin t) dt (exponentially
/// convergent for this periodic integrand; the reference for the closed-form imaginary part).
fn bessel_j_ref(n: i32, x: f64) -> f64 {
    let m = x as usize + 64;
    let h = PI_REF / m as f64;
    let mut s = 0.5 * (1.0 + (n as f64 * PI_REF).cos());
    for k in 1..m {
        let t = k as f64 * h;
        s += (n as f64 * t - x * t.sin()).cos();
    }
    s / m as f64
}

fn finite_mode(ih: i32) -> FreqMode {
    match ih {
        -2 => FreqMode::Zero,
        -1 => FreqMode::Infinite,
        _ => FreqMode::Ordinary,
    }
}

// A2 (infinite depth): wave kernel against the arbitrary-precision reference.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn green_reference() {
    let text = read_data("green_inf_reference.txt");
    let (mut worstF, mut worstX, mut worstI) = (0.0f64, 0.0f64, 0.0f64);
    let mut n = 0;
    verify_green_tables().unwrap();
    for (_, f) in data_lines(&text) {
        let (X, Y, F, FX) = (num(f[0]), num(f[1]), num(f[2]), num(f[3]));
        let w = wave_kernel(X, Y);
        let R0 = (X * X + Y * Y).sqrt();
        // scale: local magnitude of the singular parts (log term and the image 1/R0)
        let scale = 2.0 * (-Y).exp() * (Y + R0).ln().abs() + 1.0 / R0;
        let scaleX = 2.0 * (-Y).exp() / R0 + 1.0 / (R0 * R0);
        worstF = worstF.max((w.F - F).abs() / F.abs().max(scale));
        worstX = worstX.max((w.FX - FX).abs() / FX.abs().max(scaleX));
        // the imaginary part in closed form: Gi = -2 pi e^-Y J0(X), Gi_X = 2 pi e^-Y J1(X)
        let sI = 2.0 * PI_REF * (-Y).exp();
        worstI = worstI.max((w.I + sI * bessel_j_ref(0, X)).abs() / sI);
        worstI = worstI.max((w.IX - sI * bessel_j_ref(1, X)).abs() / sI);
        n += 1;
    }
    println!(
        "A2 infinite depth: {n} points, max relative error Ft {worstF:.2e}, dFt/dX {worstX:.2e} (target 2e-6), \
         Gi and Gi_X {worstI:.2e} of 2 pi e^-Y (target 1e-7)"
    );
    assert!(n > 1000, "A2 reference has the expected size");
    assert!(worstF <= 2e-6 && worstX <= 2e-6, "A2 infinite-depth Green function within 2e-6");
    assert!(worstI <= 1e-7, "A2 infinite-depth imaginary part within 1e-7 of its scale");
}

// A2 (finite depth): regular part of the kernel and its derivatives against
// the 30-digit reference, relative to max(|value|, local singular scale).
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn green_finite_reference() {
    let text = read_data("green_fin_reference.txt");
    verify_fin_tables().unwrap();
    let mut n = 0;
    let mut worst = [0.0f64; 4]; // Wr, Wr derivatives, Wi, Wi derivatives
    let mut where_: [String; 4] = Default::default();
    for (line, f) in data_lines(&text) {
        let mode: i32 = f[0].parse().unwrap();
        let (h, K, R, z, ze) = (num(f[1]), num(f[2]), num(f[3]), num(f[4]), num(f[5]));
        let r: Vec<f64> = f[6..14].iter().map(|s| num(s)).collect();
        let (scale, scaleD) = (num(f[14]), num(f[15]));
        let fm = match mode {
            0 => FreqMode::Ordinary,
            1 => FreqMode::Zero,
            _ => FreqMode::Infinite,
        };
        let fd = make_finite_depth(h, K, fm, true); // K = w^2/g as written
        let k = finite_kernel(&fd, R, z, ze, None, None);
        let got = [k.Wr, k.WrR, k.WrZ, k.Wrz, k.Wi, k.WiR, k.WiZ, k.Wiz];
        for i in 0..8 {
            let g = (if i < 4 { 0 } else { 2 }) + (if i % 4 == 0 { 0 } else { 1 });
            let e = (got[i] - r[i]).abs() / r[i].abs().max(if g % 2 != 0 { scaleD } else { scale });
            if e > worst[g] {
                worst[g] = e;
                where_[g] = format!("{} #{i}", &line[..line.len().min(60)]);
            }
        }
        n += 1;
    }
    println!(
        "A2 finite depth: {n} points, max relative error Re {:.2e}, Re' {:.2e}, Im {:.2e}, Im' {:.2e} (target 2e-6)",
        worst[0], worst[1], worst[2], worst[3]
    );
    for g in 0..4 {
        if worst[g] > 2e-6 {
            println!("  worst {g} at {}", where_[g]);
        }
    }
    assert!(n > 1500, "A2 finite-depth reference has the expected size");
    assert!(worst.iter().cloned().fold(0.0, f64::max) <= 2e-6, "A2 finite-depth Green function within 2e-6");
}

// Error map of the per-frequency tiles: tiled against
// direct evaluation of the integral form over a dense sweep of H = K h
// (1e-13 .. 1e5, the limits) and random points with R/h < 1/2, relative to
// max(|value|, local singular scale) as in A2.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn finite_tiles() {
    let mut rng = Mt64::new(17);
    let mut worst = [0.0f64; 2];
    let mut wH = [0.0f64; 2];
    let h = 1.0;
    for ih in -2..=360 {
        let fm = finite_mode(ih);
        let H = 10.0f64.powf(-13.0 + 18.0 * ih.max(0) as f64 / 360.0);
        let ft = make_finite_depth(h, H / h, fm, true);
        let fdir = make_finite_depth(h, H / h, fm, false);
        for _ in 0..60 {
            let R = 0.5 * h * rng.uniform(0.0, 1.0);
            let z = -h * rng.uniform(0.0, 1.0);
            let ze = -h * rng.uniform(0.0, 1.0);
            if R < 1e-6 && (z - ze).abs() < 1e-6 {
                continue;
            }
            let a = finite_kernel(&ft, R, z, ze, None, None);
            let b = finite_kernel(&fdir, R, z, ze, None, None);
            let v = [(z - ze).abs(), z + ze + 2.0 * h, -(z + ze), z - ze + 2.0 * h, ze - z + 2.0 * h, z + ze + 4.0 * h];
            let mut sc = 0.0;
            let mut scD = 0.0;
            for vp in v {
                sc += 1.0 / (R * R + vp * vp).sqrt();
                scD += 1.0 / (R * R + vp * vp);
            }
            let e0 = (a.Wr - b.Wr).abs() / b.Wr.abs().max(sc);
            let e1 = ((a.WrR - b.WrR).abs() / b.WrR.abs().max(scD))
                .max((a.WrZ - b.WrZ).abs() / b.WrZ.abs().max(scD))
                .max((a.Wrz - b.Wrz).abs() / b.Wrz.abs().max(scD));
            let tag = if ih < 0 { ih as f64 } else { H };
            if e0 > worst[0] {
                worst[0] = e0;
                wH[0] = tag;
            }
            if e1 > worst[1] {
                worst[1] = e1;
                wH[1] = tag;
            }
        }
    }
    println!(
        "finite-depth tiles vs direct: max relative error value {:.2e} (H = {:.3}), derivatives {:.2e} (H = {:.3}) \
         (target 5e-7)",
        worst[0], wH[0], worst[1], wH[1]
    );
    assert!(worst[0] <= 5e-7 && worst[1] <= 5e-7, "finite-depth tiles within 5e-7 of the direct evaluation");
}

// Second derivatives of the finite-depth regular kernel
// against central differences (step 1e-5 h) of the analytic first derivatives
// of the direct evaluation, both regions (R/h below and above 1/2), ordinary
// frequencies and the limits, z = zeta included (one-sided limit), relative to
// max(|value|, scale) as above.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn finite_hessian() {
    let mut rng = Mt64::new(23);
    let mut worst = 0.0f64;
    let mut same = 0.0f64;
    let mut what = "";
    let h = 1.0;
    let d = 1e-5;
    for ih in -2..=40 {
        let fm = finite_mode(ih);
        let H = 10.0f64.powf(-3.0 + 6.0 * ih.max(0) as f64 / 40.0);
        let ft = make_finite_depth(h, H / h, fm, true);
        let fdir = make_finite_depth(h, H / h, fm, false);
        for i in 0..40 {
            let R = 0.05 + 1.5 * h * rng.uniform(0.0, 1.0);
            let z = -0.05 - 0.9 * h * rng.uniform(0.0, 1.0);
            // z = zeta: a horizontal sheet
            let ze = if i % 4 == 0 { z } else { -0.05 - 0.9 * h * rng.uniform(0.0, 1.0) };
            if (R - 0.5 * h).abs() < 4.0 * d {
                continue; // the two regions meet at R = h/2
            }
            let (k, s) = finite_kernel_hess(&ft, R, z, ze, None, None);
            {
                // the value and first derivatives equal those of finite_kernel
                let k1 = finite_kernel(&ft, R, z, ze, None, None);
                let x = [k.Wr, k.WrR, k.WrZ, k.Wrz, k.Wi, k.WiR, k.WiZ, k.Wiz];
                let y = [k1.Wr, k1.WrR, k1.WrZ, k1.Wrz, k1.Wi, k1.WiR, k1.WiZ, k1.Wiz];
                for q in 0..8 {
                    same = same.max((x[q] - y[q]).abs() / 1.0f64.max(y[q].abs()));
                }
            }
            // differences at (R, zq, zeta): RR, R_R, RZ, Rz, zZ, then the imaginary parts
            let diff = |zq: f64, out: &mut [f64; 10]| {
                let mut a = finite_kernel(&fdir, R + d, zq, ze, None, None);
                let mut b = finite_kernel(&fdir, R - d, zq, ze, None, None);
                out[0] = (a.WrR - b.WrR) / (2.0 * d);
                out[2] = (a.WrZ - b.WrZ) / (2.0 * d);
                out[3] = (a.Wrz - b.Wrz) / (2.0 * d);
                out[5] = (a.WiR - b.WiR) / (2.0 * d);
                out[7] = (a.WiZ - b.WiZ) / (2.0 * d);
                out[8] = (a.Wiz - b.Wiz) / (2.0 * d);
                a = finite_kernel(&fdir, R, zq + d, ze, None, None);
                b = finite_kernel(&fdir, R, zq - d, ze, None, None);
                out[4] = (a.WrZ - b.WrZ) / (2.0 * d);
                out[9] = (a.WiZ - b.WiZ) / (2.0 * d);
                a = finite_kernel(&fdir, R, zq, ze, None, None);
                out[1] = a.WrR / R;
                out[6] = a.WiR / R;
            };
            let mut r = [0.0f64; 10];
            if z != ze {
                diff(z, &mut r);
            } else {
                // z = zeta: the direct dPhi1/dvb1 is not exactly 0 at vb1 = 0 (about 2e-7,
                // where wave_real changes branch at Y = 2H), which a difference across
                // z = zeta divides by the step; instead the limit from one side,
                // f(0) = 3 f(e) - 3 f(2e) + f(3e) + O(e^3)
                let e = 2e-4; // 20 steps d; the O(e^3) remainder at 2e-3 reached 1e-4
                let (mut f1, mut f2, mut f3) = ([0.0f64; 10], [0.0f64; 10], [0.0f64; 10]);
                diff(z + e, &mut f1);
                diff(z + 2.0 * e, &mut f2);
                diff(z + 3.0 * e, &mut f3);
                for q in 0..10 {
                    r[q] = 3.0 * f1[q] - 3.0 * f2[q] + f3[q];
                }
            }
            let mut sc = 0.0;
            let v = [(z - ze).abs(), z + ze + 2.0 * h, -(z + ze), z - ze + 2.0 * h, ze - z + 2.0 * h, z + ze + 4.0 * h];
            for vp in v {
                sc += 1.0 / (R * R + vp * vp).powf(1.5);
            }
            let got = [s.RR, s.R_R, s.RZ, s.Rz, s.zZ, s.iRR, s.iR_R, s.iRZ, s.iRz, s.izZ];
            let names = ["RR", "R_R", "RZ", "Rz", "zZ", "iRR", "iR_R", "iRZ", "iRz", "izZ"];
            for q in 0..10 {
                let e = (got[q] - r[q]).abs() / r[q].abs().max(sc);
                if e > worst {
                    worst = e;
                    what = names[q];
                }
            }
        }
    }
    println!(
        "finite-depth kernel second derivatives vs differences: max relative error {worst:.2e} ({what}) (target 1e-5)"
    );
    assert!(worst <= 1e-5, "finite-depth second derivatives within 1e-5 of the differences");
    assert!(same <= 1e-12, "finite_kernel_hess value and first derivatives equal finite_kernel");
}

// The second derivatives on and near the vertical axis: F_X / X tends to -F_YY / 2 and
// Gi_X / X to pi e^-Y (the difference is O(X^2)); the finite-depth R_R to RR. A division of the
// tables' own value on the axis (3e-8, 2e-7) by a small X made 40 % of it at X = 1e-6; the
// derivative of the direct tables' error (Y >= 2) made 3e-5 at X up to 0.1.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn axis_quotients() {
    let mut worst = 0.0f64;
    for Y in [0.5, 2.0, 2.36, 2.8, 10.0, 25.0] {
        for X in [1e-6, 1e-4, 1e-3, 1e-2, 0.05] {
            let w = wave_kernel(X, Y);
            let R0 = (X * X + Y * Y).sqrt();
            let FYY = -w.FY + 2.0 * Y / (R0 * R0 * R0);
            let (FX_X, IX_X) = wave_x_over_x(X, Y, &w, FYY);
            let eF = (FX_X + 0.5 * FYY).abs() / 1.0f64.max(FYY.abs());
            let eI = (IX_X - PI_REF * (-Y).exp()).abs() / 1.0f64.max(PI_REF * (-Y).exp());
            // the tables' accuracy and the O(X^2) difference
            worst = worst.max(eF.max(eI) / (1e-7 + X * X * 1.0f64.max(2.0 / (Y * Y * Y * Y))));
        }
    }
    let mut worst_fin = 0.0f64;
    let h = 1.0;
    for H in [0.1, 1.0, 10.0] {
        let fd = make_finite_depth(h, H / h, FreqMode::Ordinary, true);
        for R in [1e-6, 1e-4, 1e-2] {
            for z in [-0.2, -0.6] {
                let (_, s) = finite_kernel_hess(&fd, R, z, -0.4, None, None);
                let sc = 1.0 / (z + 0.4f64).abs().powf(3.0);
                worst_fin = worst_fin.max((s.R_R - s.RR).abs() / s.RR.abs().max(sc) / (1e-6 + R * R));
            }
        }
    }
    println!(
        "the quotients near the axis: max error {worst:.2e} of the bound, finite depth {worst_fin:.2e} of (1e-6 + R^2)"
    );
    assert!(worst <= 1.0, "F_X / X and Gi_X / X near the axis within the tables and O(X^2) of their limits");
    assert!(worst_fin <= 1.0, "the finite-depth R_R near the axis within 1e-6 + R^2 of RR");
}

// A3: dispersion roots against the 40-digit reference, in units of the
// double spacing at the root (full double precision: at most 2 ulp).
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn dispersion() {
    let text = read_data("dispersion_reference.txt");
    let mut n = 0;
    let mut worst = 0.0f64;
    for (_, f) in data_lines(&text) {
        let H = num(f[0]);
        let k: i32 = f[1].parse().unwrap();
        let x = num(f[2]);
        let got = if k == 0 { dispersion_x0(H) } else { dispersion_xn(H, k) };
        let ulp = f64::from_bits(x.to_bits() + 1) - x; // nextafter(x, +inf), x > 0
        worst = worst.max((got - x).abs() / ulp);
        n += 1;
    }
    println!("A3 dispersion roots: {n} roots, max error {worst:.1} ulp (target 2)");
    assert!(n > 2000, "A3 reference has the expected size");
    assert!(worst <= 2.0, "A3 dispersion roots to full double precision");
    assert!(dispersion_x0(0.0) == 0.0 && dispersion_xn(f64::INFINITY, 3) == 2.5 * PI_REF, "A3 dispersion limits");
}

// Channel kernel: Faddeeva function and lattice sum against the mpmath
// reference (scripts/gen_channel_reference.py), tiles against direct values.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn channel() {
    let text = read_data("channel_reference.txt");
    let (mut worstW, mut worstF) = (0.0f64, 0.0f64);
    let (mut nw, mut nf) = (0, 0);
    for (_, f) in data_lines(&text) {
        if f[0] == "W" {
            let (zr, zi, wr, wi) = (num(f[1]), num(f[2]), num(f[3]), num(f[4]));
            let w = faddeeva(Complex64::new(zr, zi));
            let r = Complex64::new(wr, wi);
            worstW = worstW.max((w - r).norm() / r.norm());
            nw += 1;
        } else if f[0] == "F" {
            let (k, d, x, s, fr, fi) = (num(f[1]), num(f[2]), num(f[3]), num(f[4]), num(f[5]), num(f[6]));
            let v = channel_lattice(x, s, k, d).unwrap();
            let r = Complex64::new(fr, fi);
            let ef = (v - r).norm() / 1.0f64.max(r.norm());
            if ef > 1e-10 {
                println!("  lattice k={k} x={x} s={s}: {ef:.3e} ({} {} vs {fr} {fi})", v.re, v.im);
            }
            worstF = worstF.max(ef);
            nf += 1;
        }
    }
    // tiles of Q_even, Q_odd against direct evaluation; derivatives by differences
    let W = 80.0;
    let k = 0.2;
    let t: ChannelTail = make_channel_tail(W, k, 300.0).unwrap();
    let (mut worstT, mut worstD) = (0.0f64, 0.0f64);
    let mut seed: u64 = 12345;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / 9007199254740992.0
    };
    let h2 = |r: f64| {
        let (j0, y0, _, _) = bessel_jy01(k * r);
        Complex64::new(j0, -y0)
    };
    let direct = |odd: bool, x: f64, u: f64| {
        if !odd {
            return channel_lattice(x, u, k, 2.0 * W).unwrap() - h2(x.hypot(u));
        }
        channel_lattice(x, u - W, k, 2.0 * W).unwrap() - h2(x.hypot(u - W)) - h2(x.hypot(u + W))
    };
    for i in 0..200 {
        let odd = i % 2 == 1;
        let x = 300.0 * rnd();
        let u = W * rnd();
        let (q, qx, qu) = t.eval(odd, x, u);
        let r = direct(odd, x, u);
        worstT = worstT.max((q - r).norm());
        let e = 1e-3;
        let dx = (direct(odd, x + e, u) - direct(odd, (x - e).abs(), u)) / (2.0 * e);
        let du = (direct(odd, x, u + e) - direct(odd, x, u - e)) / (2.0 * e);
        if x > e {
            worstD = worstD.max((qx - dx).norm() / k).max((qu - du).norm() / k);
        }
    }
    println!(
        "channel: Faddeeva {nw} values {worstW:.2e}, lattice {nf} values {worstF:.2e}, tiles {worstT:.2e}, \
         derivatives {worstD:.2e}"
    );
    assert!(nw >= 8 && nf >= 24, "channel reference has the expected size");
    assert!(worstW <= 1e-12, "Faddeeva function");
    assert!(worstF <= 1e-10, "channel lattice sum");
    assert!(worstT <= 1e-8 && worstD <= 1e-6, "channel tail tiles");
}

// The table checksums (verify_*_tables fail with ErrorKind::Internal on a mismatch) and the Bessel
// functions of the tables against the integral reference, both branches (tables, Hankel).
#[test]
fn tables_and_bessel() {
    verify_green_tables().unwrap();
    verify_fin_tables().unwrap();
    let mut worst = 0.0f64;
    let mut x = 0.01;
    while x < 200.0 {
        let (j0, y0, j1, y1) = bessel_jy01(x);
        worst = worst.max((j0 - bessel_j_ref(0, x)).abs()).max((j1 - bessel_j_ref(1, x)).abs());
        // the Wronskian J1 Y0 - J0 Y1 = 2 / (pi x)
        worst = worst.max(((j1 * y0 - j0 * y1) * x * PI_REF / 2.0 - 1.0).abs());
        x *= 1.07;
    }
    println!("Bessel J0, J1 (and the Wronskian with Y0, Y1): max error {worst:.2e}");
    assert!(worst <= 1e-12, "Bessel functions of the tables");
}

// The four-point near branch of the wave kernel (AVX2) equals the one-point kernel to the bit,
// in each of the four tiles of [0, 2)^2, at the tile edges, near the axis and next to the origin;
// four points of different tiles, outside [0, 2)^2 or at the origin are refused.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn wave_kernel_four_points() {
    let mut rng = Mt64::new(11);
    let mut n = 0;
    let edges = [0.0, 1e-12, 1e-6, 1e-3, 0.5, 1.0 - 1e-15, 1.0];
    for (ix, iy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        for case in 0..2000 {
            let mut X = [0.0; 4];
            let mut Y = [0.0; 4];
            for k in 0..4 {
                let (u, v) = if case < 49 {
                    (edges[(case + k) % 7], edges[(case / 7 + 3 * k) % 7])
                } else {
                    (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
                };
                X[k] = (ix as f64 + u).min(ix as f64 + 1.0 - 1e-15);
                Y[k] = (iy as f64 + v).min(iy as f64 + 1.0 - 1e-15);
                if X[k] + Y[k] == 0.0 {
                    X[k] = 1e-9;
                }
            }
            assert!(near4_applies(&X, &Y), "one tile: {X:?} {Y:?}");
            let w4 = wave_kernel_near4(&X, &Y);
            for k in 0..4 {
                let w = wave_kernel(X[k], Y[k]);
                let a = [w.F, w.FX, w.FY, w.I, w.IX, w.IY, w.c];
                let b = [w4[k].F, w4[k].FX, w4[k].FY, w4[k].I, w4[k].IX, w4[k].IY, w4[k].c];
                for q in 0..7 {
                    assert_eq!(a[q].to_bits(), b[q].to_bits(), "component {q} at ({}, {})", X[k], Y[k]);
                }
                n += 1;
            }
        }
    }
    let one = |x: f64, y: f64| ([0.3, 0.6, x, 0.9], [0.2, 0.4, y, 0.7]);
    let (X, Y) = one(0.5, 0.5);
    assert!(near4_applies(&X, &Y));
    for (x, y) in [(1.2, 0.5), (0.5, 1.2), (2.0, 0.5), (0.5, 2.0), (35.0, 0.5), (0.0, 0.0)] {
        let (X, Y) = one(x, y);
        assert!(!near4_applies(&X, &Y), "refused: ({x}, {y})");
    }
    println!("wave kernel, four points at a time: {n} points equal to the bit");
}

// The four-point branch of the direct tables of the wave kernel (AVX2) equals the one-point
// kernel (wave_kernel and wave_real) to the bit in every tile of [0, 30)² outside [0, 2)², at the
// tile edges; four points of different tiles, in [0, 2)² or beyond the tables are refused.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn wave_kernel_direct_four_points() {
    let mut rng = Mt64::new(19);
    let edges = [0.0, 1e-12, 1e-6, 0.5, 1.0 - 1e-14];
    let mut n = 0;
    for iy in 0..30 {
        for ix in 0..30 {
            if ix < 2 && iy < 2 {
                continue;
            }
            for case in 0..60 {
                let mut X = [0.0; 4];
                let mut Y = [0.0; 4];
                for k in 0..4 {
                    let (u, v) = if case < 25 {
                        (edges[(case + k) % 5], edges[(case / 5 + 2 * k) % 5])
                    } else {
                        (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
                    };
                    X[k] = ix as f64 + u;
                    Y[k] = iy as f64 + v;
                }
                assert!(direct4_applies(&X, &Y), "one tile: {X:?} {Y:?}");
                let w4 = wave_kernel_direct4(&X, &Y);
                let r4 = wave_real_direct4(&X, &Y);
                for k in 0..4 {
                    let w = wave_kernel(X[k], Y[k]);
                    let a = [w.F, w.FX, w.FY, w.I, w.IX, w.IY, w.c];
                    let b = [w4[k].F, w4[k].FX, w4[k].FY, w4[k].I, w4[k].IX, w4[k].IY, w4[k].c];
                    for q in 0..7 {
                        assert_eq!(a[q].to_bits(), b[q].to_bits(), "component {q} at ({}, {})", X[k], Y[k]);
                    }
                    let (F, FX, FY) = wave_real(X[k], Y[k]);
                    assert_eq!(
                        [F.to_bits(), FX.to_bits(), FY.to_bits()],
                        [r4[k].0.to_bits(), r4[k].1.to_bits(), r4[k].2.to_bits()]
                    );
                    n += 1;
                }
            }
        }
    }
    let one = |x: f64, y: f64| ([5.3, 5.6, x, 5.9], [2.2, 2.4, y, 2.7]);
    assert!(direct4_applies(&one(5.5, 2.5).0, &one(5.5, 2.5).1));
    for (x, y) in [(6.2, 2.5), (5.5, 3.2), (1.5, 1.5), (5.5, 30.0), (35.0, 2.5)] {
        let (X, Y) = one(x, y);
        assert!(!direct4_applies(&X, &Y), "refused: ({x}, {y})");
    }
    println!("wave kernel, direct tables four points at a time: {n} points equal to the bit");
}

// J0 and J1, and J0, Y0, J1, Y1, four points at a time (AVX2) equal the one-point evaluation to the bit in tiles of
// the tables, at their edges; four points of different tiles or beyond the tables are refused.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn bessel_four_points() {
    let mut rng = Mt64::new(13);
    let edges = [0.0, 1e-12, 1e-6, 0.5, 1.0 - 1e-15];
    let mut n = 0;
    for i in 0..30 {
        for case in 0..500 {
            let X: [f64; 4] = std::array::from_fn(|k| {
                let u = if case < 25 { edges[(case + 2 * k) % 5] } else { rng.uniform(0.0, 1.0) };
                (i as f64 + u).min(i as f64 + 1.0 - 1e-14)
            });
            let (j0, j1) = bessel_j01_4(&X).expect("one tile");
            let jy = bessel_jy01_4(&X.map(|x| x.max(1e-300))).expect("one tile");
            for k in 0..4 {
                let (a, b) = bessel_j01(X[k]);
                assert_eq!(a.to_bits(), j0[k].to_bits(), "J0 at {}", X[k]);
                assert_eq!(b.to_bits(), j1[k].to_bits(), "J1 at {}", X[k]);
                let (a, b, c, d) = bessel_jy01(X[k].max(1e-300));
                let (e, f, g, q) = jy[k];
                assert_eq!([a, b, c, d].map(f64::to_bits), [e, f, g, q].map(f64::to_bits), "J, Y at {}", X[k]);
                n += 1;
            }
        }
    }
    for x in [1.5, 30.0, 45.0] {
        assert!(bessel_j01_4(&[0.2, 0.4, x, 0.9]).is_none(), "refused: {x}");
        assert!(bessel_jy01_4(&[0.2, 0.4, x, 0.9]).is_none(), "refused: {x}");
    }
    assert!(bessel_jy01_4(&[0.2, 0.4, 0.0, 0.9]).is_none(), "refused: 0");
    println!("Bessel J0, J1, four points at a time: {n} points equal to the bit");
}

// The finite-depth kernel four pairs at a time (AVX2: tiles, near branch of the wave kernel, J0
// and J1; series form: the scaled K0, K1 and J0, Y0, J1, Y1) equals finite_kernel with the
// fin_point data to the bit, every component, at ordinary frequencies and in both limits, in both
// forms, for groups within one tile and spread over many, with the sources at the field point's
// depth, on the axis and at the coincident point on z = 0; groups across R = h/2 are refused.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn finite_kernel_four_points() {
    let mut rng = Mt64::new(17);
    let mut n = 0;
    for (h, K, mode) in [
        (3.0, 0.1, FreqMode::Ordinary),
        (3.0, 0.4, FreqMode::Ordinary),
        (2.0, 2.0, FreqMode::Ordinary),
        (450.0, 0.004, FreqMode::Ordinary),
        (50.0, 0.4, FreqMode::Ordinary),
        (1.0, 1e-13, FreqMode::Ordinary),
        (3.0, 0.0, FreqMode::Zero),
        (3.0, 0.0, FreqMode::Infinite),
    ] {
        let fd = make_finite_depth(h, K, mode, true);
        for case in 0..6000 {
            // the spread of the group: within one tile (mostly), a few tiles, the whole range;
            // the integral form (even cases) and the series form, R/h in [1/2, 6) (odd cases)
            let spread = [1e-4, 1e-2, 1.0][(case / 2) % 3];
            let series = case % 2 == 1;
            let (lo, hi) = if series { (0.5 * h, 6.0 * h) } else { (0.0, 0.5 * h * (1.0 - 1e-15)) };
            let (R0, z, ze0) = (rng.uniform(lo, hi), -h * rng.uniform(0.0, 1.0), -h * rng.uniform(0.0, 1.0));
            let mut R = [0.0; 4];
            let mut ze = [0.0; 4];
            for k in 0..4 {
                R[k] = (R0 + spread * h * rng.uniform(-0.5, 0.5)).clamp(lo, hi);
                ze[k] = (ze0 + spread * h * rng.uniform(-0.5, 0.5)).clamp(-h, 0.0);
            }
            let z = match if series { 99 } else { (case / 2) % 23 } {
                0 => {
                    R[1] = 0.0; // on the axis
                    z
                }
                1 => {
                    ze[2] = z; // at the field point's depth
                    z
                }
                2 => {
                    R[3] = 0.0; // the coincident point on z = 0
                    ze[3] = 0.0;
                    0.0
                }
                3 => 0.0,
                _ => z,
            };
            assert!(fin4_applies(&fd, &R));
            let pz = fin_point(&fd, z);
            let pze: [_; 4] = std::array::from_fn(|k| fin_point(&fd, ze[k]));
            let w4 = finite_kernel4(&fd, &R, z, &ze, &pz, std::array::from_fn(|k| &pze[k]));
            for k in 0..4 {
                let w = finite_kernel(&fd, R[k], z, ze[k], Some(&pz), Some(&pze[k]));
                let a = [w.Wr, w.WrR, w.WrZ, w.Wrz, w.Wi, w.WiR, w.WiZ, w.Wiz];
                let b = [w4[k].Wr, w4[k].WrR, w4[k].WrZ, w4[k].Wrz, w4[k].Wi, w4[k].WiR, w4[k].WiZ, w4[k].Wiz];
                for q in 0..8 {
                    assert_eq!(
                        a[q].to_bits(),
                        b[q].to_bits(),
                        "component {q}, h {h}, K {K}: R {}, z {z}, zeta {}",
                        R[k],
                        ze[k]
                    );
                }
                n += 1;
            }
        }
        assert!(!fin4_applies(&fd, &[0.1, 0.2, 0.5 * h, 0.3]));
    }
    assert!(!fin4_applies(&make_finite_depth(3.0, 0.1, FreqMode::Ordinary, false), &[0.1; 4]));
    println!("finite-depth kernel, four pairs at a time: {n} pairs equal to the bit");
}

// Informational timing of one wave-kernel record (value and derivatives), HB_TIMING only.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn time_wave_kernel(what: &str, x0: f64, x1: f64, y0: f64, y1: f64) {
    let mut rng = Mt64::new(7);
    let n = 1 << 16;
    let mut X = vec![0.0; n];
    let mut Y = vec![0.0; n];
    for i in 0..n {
        X[i] = rng.uniform(x0, x1);
        Y[i] = rng.uniform(y0, y1);
    }
    let mut sink = 0.0;
    let reps = 16;
    let t0 = Instant::now();
    for _ in 0..reps {
        for i in 0..n {
            let w = wave_kernel(X[i], Y[i]);
            sink += w.F + w.FX;
        }
    }
    let ns = t0.elapsed().as_nanos() as f64 / (reps * n) as f64;
    println!("wave kernel, {what}: {ns:.1} ns per record (checksum {sink:.3})");
}

// Informational timing of the finite-depth kernel (a TLP case: depth 450 m,
// pairs within 100 m horizontally and 30 m of the surface; and R/h >= 1/2), HB_TIMING only.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn time_finite_kernel() {
    let mut rng = Mt64::new(5);
    let n = 1 << 16;
    let (mut R, mut z, mut ze) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut sink = 0.0;
    for K in [0.004, 0.05, 0.4] {
        let fd = make_finite_depth(450.0, K, FreqMode::Ordinary, true);
        for far in 0..2 {
            for i in 0..n {
                R[i] = if far != 0 { 225.0 + 400.0 * rng.uniform(0.0, 1.0) } else { 100.0 * rng.uniform(0.0, 1.0) };
                z[i] = -30.0 * rng.uniform(0.0, 1.0);
                ze[i] = -30.0 * rng.uniform(0.0, 1.0);
            }
            let t0 = Instant::now();
            for i in 0..n {
                let k = finite_kernel(&fd, R[i], z[i], ze[i], None, None);
                sink += k.Wr + k.WrZ + k.Wi;
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            println!(
                "finite-depth kernel, K h = {:.1}, {}: {ns:.1} ns per record (checksum {sink:.3e})",
                K * 450.0,
                if far != 0 { "R/h >= 1/2 (series)" } else { "R < 100 m (integral form)" }
            );
            {
                // with the fin_point data, one at a time and four at a time (groups of nearby
                // pairs, as consecutive panels give them)
                let pz = fin_point(&fd, -10.0);
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| (R[a] as i32, ze[a] as i32).cmp(&(R[b] as i32, ze[b] as i32)));
                let (Rs, zs): (Vec<f64>, Vec<f64>) = idx.iter().map(|&i| (R[i], ze[i])).unzip();
                let pze: Vec<_> = zs.iter().map(|&s| fin_point(&fd, s)).collect();
                let t0 = Instant::now();
                for i in 0..n {
                    let k = finite_kernel(&fd, Rs[i], -10.0, zs[i], Some(&pz), Some(&pze[i]));
                    sink += k.Wr + k.WrZ + k.Wi;
                }
                let one = t0.elapsed().as_nanos() as f64 / n as f64;
                let t0 = Instant::now();
                for i in (0..n).step_by(4) {
                    let r: [f64; 4] = Rs[i..i + 4].try_into().unwrap();
                    let s: [f64; 4] = zs[i..i + 4].try_into().unwrap();
                    for k in finite_kernel4(&fd, &r, -10.0, &s, &pz, std::array::from_fn(|k| &pze[i + k])) {
                        sink += k.Wr + k.WrZ + k.Wi;
                    }
                }
                let four = t0.elapsed().as_nanos() as f64 / n as f64;
                println!(
                    "  with the depth factors: {one:.1} ns per record, four at a time {four:.1} ns (checksum {sink:.3e})"
                );
            }
        }
    }
}

#[test]
fn kernel_timings() {
    if std::env::var_os("HB_TIMING").is_none() {
        return;
    }
    time_finite_kernel();
    time_wave_kernel("tables X, Y in [0, 30]", 0.0, 30.0, 1e-3, 30.0);
    time_wave_kernel("far X in [30, 300], Y in [0, 30]", 30.0, 300.0, 0.0, 30.0);
    time_wave_kernel("deep Y in [30, 100]", 0.0, 100.0, 30.0, 100.0);
    time_wave_kernel_near4();
}

// Informational timing of the near branch (X, Y in [0, 1)) one point and four points at a time,
// HB_TIMING only.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn time_wave_kernel_near4() {
    let mut rng = Mt64::new(7);
    let n = 1 << 16;
    let X: Vec<f64> = (0..n).map(|_| rng.uniform(1e-3, 1.0)).collect();
    let Y: Vec<f64> = (0..n).map(|_| rng.uniform(1e-3, 1.0)).collect();
    let reps = 16;
    let mut sink = 0.0;
    let t0 = Instant::now();
    for _ in 0..reps {
        for i in 0..n {
            let w = wave_kernel(X[i], Y[i]);
            sink += w.F + w.FX;
        }
    }
    let one = t0.elapsed().as_nanos() as f64 / (reps * n) as f64;
    let t0 = Instant::now();
    for _ in 0..reps {
        for i in (0..n).step_by(4) {
            let x: [f64; 4] = X[i..i + 4].try_into().unwrap();
            let y: [f64; 4] = Y[i..i + 4].try_into().unwrap();
            for w in wave_kernel_near4(&x, &y) {
                sink += w.F + w.FX;
            }
        }
    }
    let four = t0.elapsed().as_nanos() as f64 / (reps * n) as f64;
    println!("wave kernel, X, Y in [0, 1): {one:.1} ns per record, four at a time {four:.1} ns (checksum {sink:.3})");
}
