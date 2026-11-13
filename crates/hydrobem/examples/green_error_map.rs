//! The error of the engine's free-surface Green functions at every point of the high-precision
//! references (tests/data/green_inf_reference.txt, green_fin_reference.txt; test-plan A2), one line
//! per point, for the error maps of the validation report (validation/kernels.py):
//!
//!     cargo run --release --example green_error_map -- infinite|finite
//!
//! infinite: X Y and the errors of Ft and dFt/dX relative to max(|value|, the local scale of the
//! singular parts), as the component test of A2; finite: H = K h, R/h, z/h, ζ/h and the largest
//! error of the real part, its derivatives, the imaginary part and its derivatives, relative to
//! max(|value|, the reference's scale).

use std::fs;

use hydrobem_core::internals::green_fin::{FreqMode, finite_kernel, make_finite_depth};
use hydrobem_core::internals::green_inf::wave_kernel;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

fn lines(name: &str) -> Vec<Vec<f64>> {
    let text = fs::read_to_string(format!("{DATA}/{name}")).expect("the reference file");
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().map(|s| s.parse::<f64>().expect("a number")).collect())
        .collect()
}

#[allow(non_snake_case)] // the notation of the theory documents
fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "infinite".to_string());
    if which == "infinite" {
        println!("# X Y err_Ft err_FtX");
        for f in lines("green_inf_reference.txt") {
            let (X, Y, F, FX) = (f[0], f[1], f[2], f[3]);
            let w = wave_kernel(X, Y);
            let R0 = (X * X + Y * Y).sqrt();
            let scale = 2.0 * (-Y).exp() * (Y + R0).ln().abs() + 1.0 / R0;
            let scaleX = 2.0 * (-Y).exp() / R0 + 1.0 / (R0 * R0);
            println!(
                "{X} {Y} {:.3e} {:.3e}",
                (w.F - F).abs() / F.abs().max(scale),
                (w.FX - FX).abs() / FX.abs().max(scaleX)
            );
        }
    } else {
        println!("# Kh R/h z/h zeta/h mode err_re err_re_d err_im err_im_d");
        for f in lines("green_fin_reference.txt") {
            let mode = f[0] as i32;
            let (h, K, R, z, ze) = (f[1], f[2], f[3], f[4], f[5]);
            let (scale, scaleD) = (f[14], f[15]);
            let fm = match mode {
                0 => FreqMode::Ordinary,
                1 => FreqMode::Zero,
                _ => FreqMode::Infinite,
            };
            let fd = make_finite_depth(h, K, fm, true);
            let k = finite_kernel(&fd, R, z, ze, None, None);
            let got = [k.Wr, k.WrR, k.WrZ, k.Wrz, k.Wi, k.WiR, k.WiZ, k.Wiz];
            let mut worst = [0.0f64; 4];
            for i in 0..8 {
                let g = (if i < 4 { 0 } else { 2 }) + usize::from(i % 4 != 0);
                let e = (got[i] - f[6 + i]).abs() / f[6 + i].abs().max(if g % 2 != 0 { scaleD } else { scale });
                worst[g] = worst[g].max(e);
            }
            println!(
                "{} {} {} {} {mode} {:.3e} {:.3e} {:.3e} {:.3e}",
                K * h,
                R / h,
                z / h,
                ze / h,
                worst[0],
                worst[1],
                worst[2],
                worst[3]
            );
        }
    }
}
