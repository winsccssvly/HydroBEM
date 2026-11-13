//! Mathematical constants and small helpers shared by the engine.

pub const PI: f64 = std::f64::consts::PI;
pub const GAMMA: f64 = 0.577_215_664_901_532_9; // Euler's constant (std's EGAMMA is not stable)
pub const LOG2: f64 = std::f64::consts::LN_2;
pub const STANDARD_GRAVITY: f64 = 9.80665; // m/s^2, the default gravity (and of a GDF file)

/// FNV-1a hash of the bytes of the doubles v (checksums of the generated tables).
pub fn fnv1a(v: &[f64], mut h: u64) -> u64 {
    for x in v {
        for byte in x.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001B3);
        }
    }
    h
}

/// A number with 17 significant digits (round trip), for messages: the %g format of precision 17
/// (fixed or exponent, trailing zeros removed).
pub fn fmt(v: f64) -> String {
    fmt_g(v, 17)
}

/// The C `%.<precision>g` format of v.
pub fn fmt_g(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = precision.max(1);
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // the exponent of v rounded to p significant digits
    let e_text = format!("{:.*e}", p - 1, v);
    let (mantissa, exp) = e_text.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    if x < -4 || x >= p as i32 {
        let m = strip_zeros(mantissa);
        let sign = if x < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", x.abs())
    } else {
        let decimals = (p as i32 - 1 - x).max(0) as usize;
        strip_zeros(&format!("{:.*}", decimals, v)).to_string()
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}
