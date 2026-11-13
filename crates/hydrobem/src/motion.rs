//! The rigid-body motion matrices (theory §3.1) and the motions of one frequency.
//!
//! Rigid-body motion matrices (theory §3.1): inertia, hydrostatic restoring with weight and the
//! additional matrices, transported to each body's rotation centre.

use crate::constants::fmt;
use crate::error::{Error, Result};
use crate::geometry::Vec3;
use crate::hydrostatics::rigid_inertia;
use crate::linalg::lu_solve;
use crate::low_order::{Output, Plan};
use crate::model_codes::{Excitation, Origin};
use crate::problem::{Body, Problem};
use crate::progress::Work;
use crate::solver::MotionMatrices;
use num_complex::{Complex32, Complex64};

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn transport(P: &Vec3, R: &Vec3) -> [f64; 36] {
    // T = [[E, -S(P - R)], [0, E]]: rigid displacement at P from the motion about R
    let d = *P - *R;
    let mut T = [0.0f64; 36];
    for i in 0..6 {
        T[i * 6 + i] = 1.0;
    }
    let S = [0.0, -d[2], d[1], d[2], 0.0, -d[0], -d[1], d[0], 0.0];
    for i in 0..3 {
        for j in 0..3 {
            T[i * 6 + 3 + j] = -S[3 * i + j];
        }
    }
    T
}

fn origin_point(b: &Body, kind: Origin, pt: &Vec3) -> Vec3 {
    if kind == Origin::CenterOfGravity {
        return b.cg;
    }
    if kind == Origin::RotationCenter {
        return b.rotation_center;
    }
    *pt
}

// Adds T_b^T K T_j for every column block of an additional matrix of body b.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn add_transported(m: &Problem, b: usize, K: &[f64], cols: i32, stiffness: bool, out: &mut [f64]) {
    let n6 = 6 * m.bodies.len();
    let cols = cols as usize;
    let bb = &m.bodies[b];
    let origin = |x: &Body| {
        if stiffness {
            origin_point(x, x.stiffness_origin, &x.stiffness_point)
        } else {
            origin_point(x, x.damping_origin, &x.damping_point)
        }
    };
    let Tb = transport(&origin(bb), &bb.rotation_center);
    let blocks = cols / 6;
    for jb in 0..blocks {
        let j = if cols == 6 { b } else { jb };
        let Tj = transport(&origin(&m.bodies[j]), &m.bodies[j].rotation_center);
        let mut KT = [0.0f64; 36];
        for r in 0..6 {
            for c in 0..6 {
                for k in 0..6 {
                    KT[r * 6 + c] += K[r * cols + jb * 6 + k] * Tj[k * 6 + c];
                }
            }
        }
        for r in 0..6 {
            for c in 0..6 {
                let mut v = 0.0;
                for k in 0..6 {
                    v += Tb[k * 6 + r] * KT[k * 6 + c];
                }
                out[(6 * b + r) * n6 + 6 * j + c] += v;
            }
        }
    }
}

/// The motion matrices of all bodies (inertia from rigid_inertia() with the mass in use).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn motion_matrices(m: &Problem) -> MotionMatrices {
    let nb = m.bodies.len();
    let n6 = 6 * nb;
    let mut out = MotionMatrices { ready: true, M: vec![0.0; n6 * n6], C: vec![0.0; n6 * n6], B: vec![0.0; n6 * n6] };
    for b in 0..nb {
        let body = &m.bodies[b];
        let h = m.hydrostatics(b as i32);
        let ready = body.has_cg && body.mass.is_some() && body.has_inertia;
        out.ready = out.ready && ready;
        if ready {
            let M = rigid_inertia(body, h.mass);
            for i in 0..6 {
                for j in 0..6 {
                    out.M[(6 * b + i) * n6 + 6 * b + j] = M[i * 6 + j];
                }
            }
        }
        for i in 0..6 {
            for j in 0..6 {
                out.C[(6 * b + i) * n6 + 6 * b + j] = h.c[i * 6 + j];
            }
        }
        if !body.stiffness.is_empty() {
            add_transported(m, b, &body.stiffness, body.stiffness_cols, true, &mut out.C);
        }
        if !body.damping.is_empty() {
            add_transported(m, b, &body.damping, body.damping_cols, false, &mut out.B);
        }
    }
    out
}

/// The motions of one ordinary frequency (theory §3.1) from A, B, X or the Haskind force of the
/// solved columns in its outputs `out` (of one frequency); nothing without rao or headings.
/// work: the cancel check and progress of the LU.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn solve_motions(m: &Problem, pl: &Plan, w: f64, nh: i32, out: &mut Output, work: Work<'_>) -> Result<()> {
    let n = pl.rad.len();
    let D = pl.D as usize;
    if !pl.rao || nh == 0 || n == 0 {
        return Ok(());
    }
    let nhu = nh as usize;
    // (C - w^2 (M + A) + i w (B + B_ext)) xi = X on the solved columns; the right-hand side
    // of a body with excitation_method = haskind is its Haskind force
    let mut Z = vec![Complex64::new(0.0, 0.0); n * n];
    let mut rhs = vec![Complex64::new(0.0, 0.0); n * nhu];
    for r in 0..n {
        for c in 0..n {
            let i = pl.rad[r].coord as usize;
            let j = pl.rad[c].coord as usize;
            let o = i * n + c;
            Z[r + c * n] = pl.CC[i * D + j] - w * w * (pl.MM[i * D + j] + out.A[o] as f64)
                + Complex64::new(0.0, w) * (out.B[o] as f64 + pl.BB[i * D + j]);
        }
    }
    for h in 0..nhu {
        for r in 0..n {
            let haskind = m.bodies[pl.rad[r].body as usize].excitation == Excitation::Haskind;
            let x: Complex32 = if haskind { out.HK[r * nhu + h] } else { out.X[h * D + pl.rad[r].coord as usize] };
            rhs[r + h * n] = Complex64::new(x.re as f64, x.im as f64);
        }
    }
    if let Err(e) = lu_solve(n as i32, &mut Z, nh, &mut rhs, work) {
        return Err(Error::new(e.kind(), format!("motion equation at omega = {}: {}", fmt(w), e.message())));
    }
    for h in 0..nhu {
        for r in 0..n {
            let v = rhs[r + h * n];
            out.M[h * D + pl.rad[r].coord as usize] = Complex32::new(v.re as f32, v.im as f32);
        }
    }
    Ok(())
}
