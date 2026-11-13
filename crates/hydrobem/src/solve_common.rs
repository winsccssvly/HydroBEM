//! The set-up shared by both methods: the plan of a solve (frequencies, headings, outputs), its
//! coordinates, the restoring of the pressure and generalized coordinates and the damping of the
//! pressure coordinates (theory §9.3), the motion matrices (theory §3.1) and the kernel data of a
//! frequency.

use crate::constants::PI;
use crate::dispersion::dispersion_x0;
use crate::error::Result;
use crate::geometry::Vec3;
use crate::green_channel::make_channel_tail;
use crate::green_fin::{FreqMode, SERIES_STOP, make_finite_depth};
use crate::low_order::{Column, FreqCtx, Geometry, Plan, PressSurf, press_shape};
use crate::model_codes::{Formulation, PressureShape, QtfHeadings};
use crate::motion::motion_matrices;
use crate::problem::Problem;
use crate::response::auto_kochin_angles;
use num_complex::Complex64;

/// The extent in x of the panel centroids, the points and the control-surface points (a channel), else 0.
pub fn channel_extent(g: &Geometry, points: &[Vec3]) -> f64 {
    if g.W <= 0.0 {
        return 0.0;
    }
    let mut lo: f64 = 1e300;
    let mut hi: f64 = -1e300;
    for p in &g.pg {
        lo = lo.min(p.c[0]);
        hi = hi.max(p.c[0]);
    }
    for x in points {
        lo = lo.min(x[0]);
        hi = hi.max(x[0]);
    }
    for c in &g.cs {
        lo = lo.min(c.x[0]);
        hi = hi.max(c.x[0]);
    }
    hi - lo
}

/// The kernel data of one frequency or limit; W the channel width (0: none),
/// xext the extent in x of the panels and points (the channel tail tiles).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn frequency_context(m: &Problem, W: f64, pl: &Plan, mode: FreqMode, w: f64, xext: f64) -> Result<FreqCtx> {
    let depth = m.env.depth;
    let mut fc = FreqCtx { mode, depth, ..Default::default() };
    if mode == FreqMode::Ordinary {
        fc.w = w;
        fc.K = w * w / pl.grav;
        fc.k = if depth.is_finite() { dispersion_x0(fc.K * depth) / depth } else { fc.K };
        fc.logc = if pl.logint { 1.0 } else { 0.0 };
        // kappa = w^2 / (g + i w eps) (theory §9.1)
        for l in &m.lids {
            fc.kappa.push(w * w / Complex64::new(pl.grav, w * l.damping_factor));
        }
    }
    if depth.is_finite() {
        fc.fd = make_finite_depth(depth, fc.K, mode, true);
    }
    if W > 0.0 && m.opt.channel_reflect != 0.0 {
        // channel: evanescent images until k1 (|m| - 1) W >= SERIES_STOP, where the
        // finite-depth series stops its evanescent terms. Without reflection (channel_reflect 0,
        // the open sea of the same depth) no wall image is summed: neither the far images nor the
        // tail are made, and no channel resonance arises
        fc.Me = 1 + (SERIES_STOP / (fc.fd.kn[0] * W)).ceil() as i32;
        if mode == FreqMode::Ordinary {
            fc.ch = make_channel_tail(W, fc.fd.k0, xext)?;
        }
    }
    Ok(fc)
}

/// The plan of a solve: sizes, constants, frequencies, headings and outputs.
pub fn base_plan(m: &Problem) -> Plan {
    let mut pl = Plan {
        D: m.coordinates(),
        rho: m.env.density,
        grav: m.env.gravity,
        diffraction: m.opt.diffraction,
        radiation: m.opt.radiation,
        rao: m.opt.rao,
        logint: m.opt.logarithmic,
        total: m.opt.formulation == Formulation::Total,
        ..Default::default()
    };
    for &w in &m.omegas {
        if w > 0.0 && w.is_finite() {
            pl.omegas.push(w);
        }
    }
    for &h in &m.headings {
        pl.heads.push(h * PI / 180.0);
    }
    pl.NK = if m.opt.drift_angles < 0 { auto_kochin_angles(m) } else { m.opt.drift_angles }; // -1: chosen
    pl.pdrift = m.opt.pressure_drift;
    pl.pqtf = if m.opt.pressure_drift && m.opt.diffraction { m.opt.pressure_qtf } else { QtfHeadings::None };
    pl.cqtf = pl.pqtf.any() && m.opt.control_surface_drift && m.opt.control_surface_qtf;
    pl.cdrift = m.opt.control_surface_drift;
    pl.fvel = m.opt.field_velocity;
    pl.bpres = m.opt.body_pressure;
    pl.bvel = m.opt.body_velocity;
    pl
}

/// The coordinates and radiation columns of the plan (rigid modes, then the pressure modes).
pub fn plan_coordinates(m: &Problem, ps: &[PressSurf], pl: &mut Plan) {
    // six rigid modes per body, its generalized modes, then its pressure modes
    let mut gm = 0;
    for b in 0..m.bodies.len() {
        let off = m.derived()[b].coordinate_offset;
        let bi = b as i32;
        for j in 1..=6 {
            pl.coords.push(Column { coord: off + j - 1, body: bi, mode: j, ..Default::default() });
        }
        for k in 0..m.bodies[b].generalized.len() {
            pl.coords.push(Column {
                coord: off + 6 + k as i32,
                body: bi,
                mode: -1,
                ps: -1,
                shape: PressureShape::Uniform,
                gm,
            });
            gm += 1;
        }
        for (q, s) in ps.iter().enumerate() {
            if s.body == bi {
                for k in 0..s.shapes.len() {
                    pl.coords.push(Column {
                        coord: s.coord0 + k as i32,
                        body: bi,
                        mode: 0,
                        ps: q as i32,
                        shape: s.shapes[k],
                        gm: -1,
                    });
                }
            }
        }
    }
    for id in m.radiation_mode_ids() {
        pl.rad.push(pl.coords[(id - 1) as usize]);
    }
}

/// The restoring of the pressure coordinates of one point (body-local lc, area) of surface s.
/// Cp: Dn x Dn row-major.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn add_pressure_point(m: &Problem, s: &PressSurf, lc: &Vec3, area: f64, Dn: i32, Cp: &mut [f64]) {
    // C_kl = -rho g int n_k n_l dS (opposite sign: numeric_compatibility), the chamber
    // coupling C_3k = rho g int n_k, C_4k = rho g int n_k y, C_5k = -rho g int n_k x in
    // the rigid rows and, unless numeric_compatibility, also in the pressure rows
    // (numeric_compatibility: the rigid rows only)
    let rg = m.env.density * m.env.gravity;
    let b = &m.bodies[s.body as usize];
    let nm = s.shapes.len() as i32;
    let rig = m.derived()[s.body as usize].coordinate_offset;
    let x = lc[0] - b.rotation_center[0];
    let y = lc[1] - b.rotation_center[1];
    let Dn = Dn as usize;
    for k in 0..nm {
        let nk = press_shape(s.shapes[k as usize], lc);
        let ck = (s.coord0 + k) as usize;
        for l in 0..nm {
            Cp[ck * Dn + (s.coord0 + l) as usize] +=
                (if s.numeric { 1.0 } else { -1.0 }) * rg * nk * press_shape(s.shapes[l as usize], lc) * area;
        }
        if !s.chamber {
            continue;
        }
        let f = [rg * nk * area, rg * nk * y * area, -rg * nk * x * area];
        for r in 0..3 {
            let rr = (rig + 2 + r) as usize;
            Cp[rr * Dn + ck] += f[r as usize];
            if !s.numeric {
                Cp[ck * Dn + rr] += f[r as usize];
            }
        }
    }
}

/// The additional stiffness and damping of the pressure modes.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn add_pressure_mode_terms(m: &Problem, ps: &[PressSurf], Dn: i32, Cp: &mut [f64], Bp: &mut [f64]) {
    // the additional stiffness and damping of each pressure mode
    let Dn = Dn as usize;
    for s in ps {
        for k in 0..s.shapes.len() {
            let c = s.coord0 as usize + k;
            let pm = &m.bodies[s.body as usize].pressure[(s.id - 1) as usize].modes[k];
            Cp[c * Dn + c] += pm.stiffness;
            Bp[c * Dn + c] += pm.damping;
        }
    }
}

/// The D x D motion matrices of the plan (with rao): the rigid matrices plus Cp, Bp.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn plan_motion_matrices(m: &Problem, Cp: &[f64], Bp: &[f64], pl: &mut Plan) {
    if !pl.rao {
        return;
    }
    let mm = motion_matrices(m);
    let Dn = pl.D as usize;
    let N6 = 6 * m.bodies.len();
    pl.MM = vec![0.0; Dn * Dn];
    pl.CC = Cp.to_vec();
    pl.BB = Bp.to_vec();
    for a in &pl.coords {
        for c in &pl.coords {
            if a.mode <= 0 || c.mode <= 0 {
                continue;
            }
            let ri = (6 * a.body + a.mode - 1) as usize;
            let rj = (6 * c.body + c.mode - 1) as usize;
            let o = a.coord as usize * Dn + c.coord as usize;
            pl.MM[o] = mm.M[ri * N6 + rj];
            pl.CC[o] += mm.C[ri * N6 + rj];
            pl.BB[o] += mm.B[ri * N6 + rj];
        }
    }
    // the mode matrices of the bodies with generalized modes (6 + G coordinates from the body's first)
    for b in 0..m.bodies.len() {
        let body = &m.bodies[b];
        let off = m.derived()[b].coordinate_offset as usize;
        let n6 = 6 + body.generalized.len();
        let mats: [(&Vec<f64>, &mut Vec<f64>); 3] =
            [(&body.mode_mass, &mut pl.MM), (&body.mode_damping, &mut pl.BB), (&body.mode_stiffness, &mut pl.CC)];
        for (src, dst) in mats {
            if !src.is_empty() {
                for i in 0..n6 {
                    for j in 0..n6 {
                        dst[(off + i) * Dn + off + j] += src[i * n6 + j];
                    }
                }
            }
        }
    }
}

/// Restoring of the generalized modes of one body at one wetted point: nr the six
/// rigid normals about R, ng and wg the normal velocity and vertical displacement of each
/// generalized mode.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn add_generalized_restoring(
    m: &Problem,
    body: i32,
    nr: &[f64; 6],
    ng: &[f64],
    area: f64,
    wg: &[f64],
    Dn: i32,
    Cp: &mut [f64],
) {
    // c_ij = rho g int n_j w_i dS for the rows i of the generalized modes, every column j of
    // the body (theory §10.1, w_i the vertical displacement of mode i; normals
    // into the body: -n_j here); the rigid rows keep their own restoring
    let b = &m.bodies[body as usize];
    let G = b.generalized.len();
    let off = m.derived()[body as usize].coordinate_offset as usize;
    let rg = m.env.density * m.env.gravity * area;
    let Dn = Dn as usize;
    for i in 6..6 + G {
        for j in 0..6 + G {
            let nj = if j < 6 { nr[j] } else { ng[j - 6] };
            Cp[(off + i) * Dn + off + j] -= rg * nj * wg[i - 6];
        }
    }
}
