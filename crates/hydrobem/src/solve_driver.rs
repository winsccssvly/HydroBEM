//! The parts of a solve that both methods share: the allocation of the
//! outputs, the loop over the limits and the ordinary frequencies (threads, progress, cancel),
//! and the results that do not depend on the method. solver.rs (low order) and
//! ho_solver.rs (higher order) build their geometry and statics, then call these.

use crate::codes::QuantityId;
use crate::constants::PI;
use num_complex::{Complex32, Complex64};
use std::sync::Mutex;

use crate::error::{ErrorKind, Location, Result, ResultExt};
use crate::geometry::Vec3;
use crate::green_fin::FreqMode;
use crate::heading_pairs::{CoordinateRows, heading_pairs, put_pairs, select_pairs};
use crate::low_order::{Output, Plan, PressSurf};
use crate::model_codes::QtfHeadings;
use crate::model_types::ProgressAt;
use crate::parallel::{FrequencyPools, StepRows};
use crate::problem::Problem;
use crate::progress::{Phase, SecondOrderStep, SolveContext, Work};
use crate::quadratic_qtf::{QtfSamples, control_surface_qtfs, quadratic_qtfs};
use crate::results::{Results, put, put_vec};
use crate::route::Route;
use crate::solve_common::add_pressure_mode_terms;
use crate::solve_common::plan_motion_matrices;
use crate::tanks::field_all;
use crate::tanks::plan_tanks;
use std::mem::take;

/// The limits of the solve: omega = 0 and omega = inf, radiation only (rules 6 and 9).
pub fn limit_modes(m: &Problem, pl: &Plan) -> Vec<FreqMode> {
    let mut limits = Vec::new();
    if pl.radiation && m.has_zero() {
        limits.push(FreqMode::Zero);
    }
    if pl.radiation && m.has_infinite() {
        limits.push(FreqMode::Infinite);
    }
    limits
}

/// Zero outputs of F frequencies (all of the plan, or one) for P field points, L lid points,
/// Q pressure-surface points and Nb body output points; the optional ones only when the plan asks
/// for them.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn allocate_outputs(m: &Problem, pl: &Plan, F: usize, P: i32, L: i32, Q: i32, Nb: i32) -> Output {
    let K = pl.rad.len();
    let D = pl.D as usize;
    let NK = pl.NK as usize;
    let Hd = if pl.diffraction { pl.heads.len() } else { 0 };
    let NB = m.bodies.len();
    let (P, L, Q, Nb) = (P as usize, L as usize, Q as usize, Nb as usize);
    let z = Complex32::new(0.0, 0.0);
    let mut out = Output::default();
    out.A = vec![0.0f32; F * D * K];
    out.B = out.A.clone();
    out.X = vec![z; F * Hd * D];
    out.FK = out.X.clone();
    out.SC = out.X.clone();
    out.M = out.X.clone();
    out.HK = vec![z; F * K * Hd];
    out.frad = vec![z; F * P * K];
    out.finc = vec![z; F * P * Hd];
    out.fsca = out.finc.clone();
    out.ftot = out.finc.clone();
    out.A0 = vec![0.0f32; D * K];
    out.Ainf = out.A0.clone();
    out.lrad = vec![z; F * L * K];
    out.ldif = vec![z; F * L * Hd];
    out.prad = vec![z; F * Q * K];
    out.pdif = vec![z; F * Q * Hd];
    out.bpr = vec![z; if pl.bpres { F * Nb * K } else { 0 }];
    out.bpd = vec![z; if pl.bpres { F * Nb * Hd } else { 0 }];
    out.bvr = vec![z; if pl.bvel { F * Nb * K * 3 } else { 0 }];
    out.bvd = vec![z; if pl.bvel { F * Nb * Hd * 3 } else { 0 }];
    out.PD = vec![0.0; if pl.pdrift { F * Hd * 9 * NB } else { 0 }];
    let GT: usize = m.bodies.iter().map(|b| b.generalized.len()).sum();
    out.PG = vec![0.0; if pl.pdrift { F * Hd * GT } else { 0 }];
    out.CD = vec![0.0; if pl.cdrift { F * Hd * 9 * NB } else { 0 }];
    out.qtf = vec![QtfSamples::default(); if pl.pqtf.any() && Hd > 0 { F } else { 0 }];
    out.fvr = vec![z; if pl.fvel { F * P * K * 3 } else { 0 }];
    out.fvi = vec![z; if pl.fvel { F * P * Hd * 3 } else { 0 }];
    out.fvs = out.fvi.clone();
    out.fvt = out.fvi.clone();
    out.kr = vec![z; F * NK * K];
    out.krt = out.kr.clone();
    out.kd = vec![z; F * NK * Hd];
    out.kdt = out.kd.clone();
    out
}

/// The bytes allocate_outputs allocates for all the frequencies of the plan (the memory estimate).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn output_bytes(m: &Problem, pl: &Plan, P: i32, L: i32, Q: i32, Nb: i32) -> i64 {
    outputs_of(m, pl, pl.omegas.len() as i64, P, L, Q, Nb)
}

/// The bytes allocate_outputs allocates for one frequency: the outputs a frequency thread fills
/// before run_frequencies writes them into those of all (the memory estimate).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn frequency_output_bytes(m: &Problem, pl: &Plan, P: i32, L: i32, Q: i32, Nb: i32) -> i64 {
    outputs_of(m, pl, 1, P, L, Q, Nb)
}

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the counts of the outputs, passed as they are
fn outputs_of(m: &Problem, pl: &Plan, F: i64, P: i32, L: i32, Q: i32, Nb: i32) -> i64 {
    // the arrays of allocate_outputs, in its order: float 4 bytes, complex64 8, double 8
    let K = pl.rad.len() as i64;
    let D = pl.D as i64;
    let NK = pl.NK as i64;
    let Hd = if pl.diffraction { pl.heads.len() as i64 } else { 0 };
    let NB = m.bodies.len() as i64;
    let (P, L, Q, Nb) = (P as i64, L as i64, Q as i64, Nb as i64);
    let mut b = 2 * F * D * K * 4; // A, B
    b += 4 * F * Hd * D * 8 + F * K * Hd * 8; // X, FK, SC, M; HK
    b += F * P * K * 8 + 3 * F * P * Hd * 8; // frad; finc, fsca, ftot
    b += 2 * D * K * 4; // A0, Ainf
    b += F * L * (K + Hd) * 8 + F * Q * (K + Hd) * 8; // lid and pressure-surface elevations
    if pl.bpres {
        b += F * Nb * (K + Hd) * 8;
    }
    if pl.bvel {
        b += 3 * F * Nb * (K + Hd) * 8;
    }
    if pl.pdrift {
        let GT: i64 = m.bodies.iter().map(|b| b.generalized.len() as i64).sum();
        b += F * Hd * (9 * NB + GT) * 8;
    }
    if pl.cdrift {
        b += F * Hd * 9 * NB * 8;
    }
    if pl.fvel {
        b += 3 * F * P * (K + 3 * Hd) * 8;
    }
    b += 2 * F * NK * (K + Hd) * 8; // Kochin functions and derivatives
    b
}

/// The bytes of the quadratic QTFs with nw points of the wetted surfaces: the first-order values of
/// every frequency and heading (and their geometry; the waterline and the tanks not counted) and
/// the coefficients (the memory estimate).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn qtf_bytes(m: &Problem, pl: &Plan, nw: i64) -> i64 {
    let F = pl.omegas.len() as i64;
    let H = if pl.diffraction { pl.heads.len() as i64 } else { 0 };
    let NB = m.bodies.len() as i64;
    let samples = F * (H * (4 * nw + 6 * NB) * 16 + nw * 8 * 8);
    let pairs = if pl.pqtf == QtfHeadings::Pairs { H } else { 1 };
    // the pairs of every heading pair, the equal headings and with "bidirectional" the pairs as given
    // (the generalized modes add their rows, GT of all bodies)
    let GT: i64 = m.bodies.iter().map(|b| b.generalized.len() as i64).sum();
    let coefficients = 2 * F * F * H * H * (6 * NB + GT) * 16 + 2 * F * F * H * pairs * (6 * NB + GT) * 16;
    // the control-surface QTFs: their coefficients (their points' values not counted)
    let cs = if pl.cqtf { coefficients } else { 0 };
    samples + coefficients + cs
}

/// The bytes of the results of a solve beyond its outputs, while they are built and after (the
/// memory estimate): (added, copy). The results take the arrays of the outputs (common_results and
/// the put_*_outputs move them, no copy); added are the quantities made anew, which stay beside them
/// (the drifts of the heading pairs, the Newman QTFs, the rows of the tank points of the field
/// quantities); copy is the largest array made from another while that is still held (the Haskind
/// excitation transposed from its columns, the QTFs of the pairs or a drift's six rows placed on the
/// coordinates, a field quantity at every field point with tanks and its split by owner, the values
/// at the specified body points, Np of them, interpolated). P the field points of the solve, as
/// output_bytes counts them.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn result_bytes(m: &Problem, pl: &Plan, P: i64, Np: i64) -> (i64, i64) {
    let F = pl.omegas.len() as i64;
    let K = pl.rad.len() as i64;
    let D = pl.D as i64;
    let Hd = if pl.diffraction { pl.heads.len() as i64 } else { 0 };
    let NB = m.bodies.len() as i64;
    let n6 = 6 * NB;
    let np = heading_pairs(Hd as usize, m.opt.drift_pairs).len() as i64;
    // the drifts (pressure, control surface, momentum from the Kochin functions): [F, np, D] and
    // their moments about the moving centres [F, np, 3 NB], complex128
    let drifts = pl.pdrift as i64 + pl.cdrift as i64 + (pl.NK > 0) as i64;
    let mut added = if Hd > 0 { drifts * F * np * (D + 3 * NB) * 16 } else { 0 };
    // the Newman QTFs [F, F, H, D], float64
    added += m.opt.newman.count() as i64 * F * F * Hd * D * 8;
    // the field quantities and the widths of their rows (K or H, 3 K or 3 H for the velocities)
    let mut widths = Vec::new();
    if m.has_field_points && m.opt.field_pressure {
        widths.extend([K, Hd, Hd, Hd]);
    }
    if m.has_field_points && pl.fvel {
        widths.extend([3 * K, 3 * Hd, 3 * Hd, 3 * Hd]);
    }
    let Pa = m.field_points.len() as i64;
    let tanks = !m.field_tanks.is_empty();
    if tanks {
        added += F * (Pa - P).max(0) * widths.iter().sum::<i64>() * 8;
    }
    let mut copy: i64 = 0;
    if pl.radiation && Hd > 0 {
        copy = copy.max(F * K * Hd * 8);
    }
    if D != n6 {
        copy = copy.max(if Hd > 0 && drifts > 0 { F * np * n6 * 16 } else { 0 });
        if pl.pqtf.any() && Hd > 0 && !m.opt.qtf_diagonal {
            let hp = heading_pairs(Hd as usize, m.opt.qtf_pairs).len() as i64;
            copy = copy.max(F * F * hp * n6 * 16);
        }
    }
    if tanks {
        copy = copy.max(F * Pa * widths.iter().copied().max().unwrap_or(0) * 8);
    }
    let body = if pl.bvel {
        3 * K.max(Hd)
    } else if pl.bpres {
        K.max(Hd)
    } else {
        0
    };
    copy = copy.max(F * Np * body * 8);
    (added, copy)
}

/// The pressure-mode stiffness and damping added to Cp and Bp, the motion matrices and the tank
/// problems of the plan (after the method has added its restoring of the pressure and
/// generalized coordinates to Cp).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn finish_plan(
    m: &Problem,
    ps: &[PressSurf],
    Cp: &mut [f64],
    Bp: &mut [f64],
    pl: &mut Plan,
    ctx: &SolveContext,
) -> Result<()> {
    add_pressure_mode_terms(m, ps, pl.D, Cp, Bp);
    plan_motion_matrices(m, Cp, Bp, pl);
    plan_tanks(m, pl, ctx)?; // the tank problems
    Ok(())
}

/// Solves the limits one after the other, then the ordinary frequencies on m.opt.frequency_threads
/// workers, each frequency on the inner pool of its worker (parallel.rs): one(mode, fi, work) for
/// each (fi = -1 for a limit) returns the outputs of its frequency (allocate_outputs of one
/// frequency), which are written into `out` at fi once it has finished. Every frequency is
/// independent, so the results do not depend on the thread counts. The reporter of the context
/// hears of every finished frequency; a cancel request or an error stops the remaining ones.
/// The phases of the solve: the limits (on the first worker), the ordinary frequencies, then the
/// second order when the plan asks for second-order loads (else the writing of the results).
/// Returns ErrorKind::Cancelled when the solve was cancelled; else the first error of a frequency.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn run_frequencies(
    m: &Problem,
    pl: &Plan,
    ctx: &SolveContext,
    out: &mut Output,
    one: &(dyn Fn(FreqMode, i32, Work<'_>) -> Result<Output> + Sync),
) -> Result<()> {
    let limits = limit_modes(m, pl);
    let F = pl.omegas.len();
    ctx.reporter.set_frequencies((F + limits.len()) as i32);
    let pools = FrequencyPools::new(m.opt.frequency_threads, m.opt.inner_threads, F)?;
    // the board of the model was made for these (Model::begin_solve)
    debug_assert!(ctx.board.as_ref().is_none_or(|b| b.slots() == limits.len() + F));
    if !limits.is_empty() {
        ctx.phase(Phase::Limits);
    }
    for (li, mode) in limits.into_iter().enumerate() {
        ctx.check_cancel()?;
        // on the inner pool of the first worker
        let work = ctx.work_at(ProgressAt::Limit, li).on(0);
        let o = tracked(ctx, work, || pools.install(|| one(mode, -1, work)))?;
        out.put_limit(mode, &o);
        ctx.frequency_done(ProgressAt::Limit, Some(0));
    }
    ctx.phase(Phase::FirstOrder);
    let out = Mutex::new(out);
    pools.for_each(F, ctx, |fi, w| {
        let at = Location { frequency: Some(fi), ..Default::default() }; // the zero-based ordinary frequency of an error
        let work = ctx.work(ProgressAt::Frequency(fi)).on(w);
        let o = tracked(ctx, work, || one(FreqMode::Ordinary, fi as i32, work).located(at))?;
        out.lock().unwrap_or_else(|p| p.into_inner()).put_frequency(fi, &o);
        ctx.frequency_done(ProgressAt::Frequency(fi), Some(w));
        Ok(())
    })?;
    ctx.check_cancel()?;
    ctx.phase(if has_second_order(m, pl) { Phase::SecondOrder } else { Phase::Finishing });
    Ok(())
}

/// Whether the solve computes second-order loads after its frequencies: the quadratic QTFs or
/// the potential QTF (the drifts are outputs of each frequency).
fn has_second_order(m: &Problem, pl: &Plan) -> bool {
    (pl.pqtf.any() && pl.diffraction && !pl.heads.is_empty()) || m.opt.potential_qtf.any()
}

// a frequency run with its slot on the board of the solve, on its worker: begun, then done,
// failed or cancelled
fn tracked(ctx: &SolveContext, work: Work, f: impl FnOnce() -> Result<Output>) -> Result<Output> {
    let (Some(b), Some(s)) = (&ctx.board, work.slot()) else { return f() };
    b.begin(s, work.worker());
    let r = f();
    match &r {
        Ok(_) => b.finish_slot(s),
        Err(e) => b.stop_slot(s, e.kind() == ErrorKind::Cancelled),
    }
    r
}

/// The quadratic QTFs of every body, [..][6 NB] with the moments about the body origin O (body
/// axes), turned to the moments about each body's rotation centre R: M_R = M_O − R × F (theory §1.1;
/// R body-fixed, the moments in the body's mean axes: the transfer is exact for every pair). The
/// bodies with R = 0 keep their values as they are.
fn moments_about_rotation_centers(m: &Problem, v: &mut [Complex64]) {
    let n6 = 6 * m.bodies.len();
    for (bi, b) in m.bodies.iter().enumerate() {
        let r = b.rotation_center;
        if r == Vec3::ZERO {
            continue;
        }
        for row in v.chunks_exact_mut(n6) {
            let f = &mut row[6 * bi..6 * bi + 6];
            let (fx, fy, fz) = (f[0], f[1], f[2]);
            f[3] -= r[1] * fz - r[2] * fy;
            f[4] -= r[2] * fx - r[0] * fz;
            f[5] -= r[0] * fy - r[1] * fx;
        }
    }
}

/// The results that do not depend on the method: the frequencies, headings and radiation ids,
/// the forces and limits, the motions, the drifts, the Kochin functions, the mode matrices
/// (mode_matrices: pressure or generalized coordinates) and the field outputs. The quadratic
/// QTFs are steps of the second order of the solve (ctx: their progress and cancel request), their
/// rows ω_a on the frequency workers. An internal error when the threads of the QTFs cannot start;
/// ErrorKind::Cancelled when the solve was cancelled.
#[allow(clippy::too_many_arguments)] // the results of the solve, its matrices and its context
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn common_results(
    m: &Problem,
    pl: &Plan,
    route: &Route,
    out: &mut Output,
    Cp: &[f64],
    Bp: &[f64],
    mode_matrices: bool,
    ctx: &SolveContext,
) -> Result<Box<Results>> {
    let F = pl.omegas.len() as i64;
    let H = pl.heads.len() as i64;
    let K = pl.rad.len() as i64;
    let D = pl.D as i64;
    let NB = m.bodies.len() as i64;
    let mut r = Box::new(Results { route: route.name(), ..Default::default() });
    let ids: Vec<i32> = pl.rad.iter().map(|c| c.coord + 1).collect();
    put(&mut r, QuantityId::Omegas, vec![F], &pl.omegas);
    put(&mut r, QuantityId::Headings, vec![m.headings.len() as i64], &m.headings);
    put(&mut r, QuantityId::RadiationModeIds, vec![K], &ids);
    if pl.radiation && m.has_zero() {
        put_vec(&mut r, QuantityId::AddedMassZeroFrequency, vec![D, K], take(&mut out.A0));
    }
    if pl.radiation && m.has_infinite() {
        put_vec(&mut r, QuantityId::AddedMassInfiniteFrequency, vec![D, K], take(&mut out.Ainf));
    }
    if pl.radiation {
        put_vec(&mut r, QuantityId::AddedMass, vec![F, D, K], take(&mut out.A));
        put_vec(&mut r, QuantityId::RadiationDamping, vec![F, D, K], take(&mut out.B));
    }
    if pl.diffraction {
        put_vec(&mut r, QuantityId::ExcitationDiffraction, vec![F, H, D], take(&mut out.X));
        put_vec(&mut r, QuantityId::ExcitationFroudeKrylov, vec![F, H, D], take(&mut out.FK));
        put_vec(&mut r, QuantityId::ExcitationScattering, vec![F, H, D], take(&mut out.SC));
        // (no Haskind relation with a free damper)
        let mut free_damper = false;
        for b in &m.bodies {
            free_damper = free_damper || b.has_free_damper();
        }
        if pl.radiation && !free_damper {
            // computed per radiation column [F, K, H]; given as the other exciting forces, [F, H, K]
            let mut hk = vec![Complex32::new(0.0, 0.0); out.HK.len()];
            for f in 0..F {
                for k in 0..K {
                    for h in 0..H {
                        hk[((f * H + h) * K + k) as usize] = out.HK[((f * K + k) * H + h) as usize];
                    }
                }
            }
            out.HK = Vec::new(); // freed before the copy is moved into the results
            put_vec(&mut r, QuantityId::ExcitationHaskind, vec![F, H, K], hk);
        }
        if pl.rao {
            put_vec(&mut r, QuantityId::Motion, vec![F, H, D], take(&mut out.M));
        }
        // the quadratic QTFs of every heading pair (theory §12.1), with qtf_diagonal at equal
        // frequencies only (computed for the bidirectional drifts alone)
        let n6 = 6 * NB;
        let (Fu, Hu, n6u, NBu) = (F as usize, H as usize, n6 as usize, NB as usize);
        let mut qp: Option<(Vec<Complex64>, Vec<Complex64>)> = None;
        let mut qc: Option<(Vec<Complex64>, Vec<Complex64>)> = None;
        // the generalized forces of the generalized modes (their coordinates); the loads are put on
        // the coordinate axis, the rigid rows and the generalized forces at their coordinates
        let rows = CoordinateRows::of(m);
        let GT = rows.generalized.len();
        let mut qg: Option<(Vec<Complex64>, Vec<Complex64>)> = None;
        if pl.pqtf.any() && H > 0 {
            let (all, diag) = (pl.pqtf == QtfHeadings::Pairs, m.opt.qtf_diagonal);
            let gm: Vec<usize> = m.bodies.iter().map(|b| b.generalized.len()).collect();
            // the rows of the pairs on the frequency workers of the solve, each with its inner
            // threads (a pool that cannot start is an error)
            let (ft, it) = (m.opt.frequency_threads, m.opt.inner_threads);
            let rows = StepRows::new(ctx.step(SecondOrderStep::QuadraticPressure, Fu), ft, it, Fu)?;
            let (mut d6, mut s6, dg, sg) = quadratic_qtfs(&rows, &out.qtf, Hu, &gm, pl.grav, all, diag);
            rows.finish()?;
            moments_about_rotation_centers(m, &mut d6);
            moments_about_rotation_centers(m, &mut s6);
            qp = Some((d6, s6));
            if GT > 0 {
                qg = Some((dg, sg));
            }
            if pl.cqtf {
                let rows = StepRows::new(ctx.step(SecondOrderStep::QuadraticControlSurface, Fu), ft, it, Fu)?;
                let (mut d6, mut s6) = control_surface_qtfs(&rows, &out.qtf, Hu, NBu, pl.grav, all, diag);
                rows.finish()?;
                moments_about_rotation_centers(m, &mut d6);
                moments_about_rotation_centers(m, &mut s6);
                qc = Some((d6, s6));
            }
            out.qtf = Vec::new(); // the first-order values of the QTFs, no longer needed
        }
        // the drifts of the heading pairs: computed per body as nine values (the force and moment
        // about its rotation centre R, the moment about the moving R) of the equal headings; another
        // pair the diagonal of the difference QTFs of the heading pairs, without the moving moment (NaN)
        let dp = heading_pairs(Hu, m.opt.drift_pairs);
        let np = dp.len() as i64;
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let drift = |r: &mut Results, d9: &[f64], full: Option<&Vec<Complex64>>, q3: QuantityId| {
            let mut six = Vec::with_capacity(Fu * dp.len() * n6u);
            let mut three = Vec::with_capacity(Fu * dp.len() * 3 * NBu);
            for f in 0..Fu {
                for &(a, b) in &dp {
                    for bi in 0..NBu {
                        if a == b {
                            let k = ((f * Hu + a) * NBu + bi) * 9;
                            six.extend(d9[k..k + 6].iter().map(|&x| Complex64::new(x, 0.0)));
                            three.extend(d9[k + 6..k + 9].iter().map(|&x| Complex64::new(x, 0.0)));
                        } else {
                            let k = (((f * Fu + f) * Hu + a) * Hu + b) * n6u + 6 * bi;
                            six.extend((0..6).map(|c| full.map_or(nan, |v| v[k + c])));
                            three.extend([nan; 3]);
                        }
                    }
                }
            }
            put_vec(r, q3, vec![F, np, 3 * NB], three);
            six
        };
        if pl.pdrift {
            let six = drift(&mut r, &out.PD, qp.as_ref().map(|q| &q.0), QuantityId::DriftPressureIntegrationBodyAxes);
            // the generalized forces of the drift: the equal headings computed, another pair the
            // diagonal of the generalized difference QTFs
            let full = qg.as_ref().map(|q| &q.0);
            let mut v = Vec::with_capacity(Fu * dp.len() * GT);
            if GT > 0 {
                for f in 0..Fu {
                    for &(a, b) in &dp {
                        if a == b {
                            let k = (f * Hu + a) * GT;
                            v.extend(out.PG[k..k + GT].iter().map(|&x| Complex64::new(x, 0.0)));
                        } else {
                            let k = (((f * Fu + f) * Hu + a) * Hu + b) * GT;
                            v.extend((0..GT).map(|c| full.map_or(nan, |q| q[k + c])));
                        }
                    }
                }
            }
            put_vec(&mut r, QuantityId::DriftPressureIntegration, vec![F, np, D], rows.place(six, &v));
        }
        if pl.cdrift {
            let six = drift(&mut r, &out.CD, qc.as_ref().map(|q| &q.0), QuantityId::DriftControlSurfaceBodyAxes);
            put_vec(&mut r, QuantityId::DriftControlSurface, vec![F, np, D], rows.place(six, &[]));
        }
        if pl.pdrift || pl.cdrift {
            put_pairs(&mut r, QuantityId::DriftHeadingPairs, &m.headings, &dp);
        }
        // the QTFs of the heading pairs asked for
        if !m.opt.qtf_diagonal && (qp.is_some() || qc.is_some()) {
            let hp = heading_pairs(Hu, m.opt.qtf_pairs);
            let dims = vec![F, F, hp.len() as i64, D];
            let pick = |v: &[Complex64]| select_pairs(v, Fu * Fu, Hu, n6u, &hp);
            let pickg = |v: Option<&Vec<Complex64>>| v.map_or(Vec::new(), |v| select_pairs(v, Fu * Fu, Hu, GT, &hp));
            if let Some((dif, sum)) = &qp {
                let (gd, gs) = (pickg(qg.as_ref().map(|q| &q.0)), pickg(qg.as_ref().map(|q| &q.1)));
                put_vec(
                    &mut r,
                    QuantityId::QtfQuadraticPressureIntegrationDifference,
                    dims.clone(),
                    rows.place(pick(dif), &gd),
                );
                put_vec(
                    &mut r,
                    QuantityId::QtfQuadraticPressureIntegrationSum,
                    dims.clone(),
                    rows.place(pick(sum), &gs),
                );
            }
            if let Some((dif, sum)) = &qc {
                put_vec(
                    &mut r,
                    QuantityId::QtfQuadraticControlSurfaceDifference,
                    dims.clone(),
                    rows.place(pick(dif), &[]),
                );
                put_vec(&mut r, QuantityId::QtfQuadraticControlSurfaceSum, dims, rows.place(pick(sum), &[]));
            }
            put_pairs(&mut r, QuantityId::QtfHeadingPairs, &m.headings, &hp);
        }
        if pl.NK > 0 {
            // the Kochin functions of the mean drift (theory §11.1)
            let NK = pl.NK as i64;
            let th: Vec<f64> = (0..pl.NK).map(|t| 2.0 * PI * t as f64 / pl.NK as f64).collect();
            put(&mut r, QuantityId::KochinAngles, vec![NK], &th);
            if K > 0 {
                put_vec(&mut r, QuantityId::KochinRadiation, vec![F, NK, K], take(&mut out.kr));
                put_vec(&mut r, QuantityId::KochinRadiationDerivative, vec![F, NK, K], take(&mut out.krt));
            }
            put_vec(&mut r, QuantityId::KochinDiffraction, vec![F, NK, H], take(&mut out.kd));
            put_vec(&mut r, QuantityId::KochinDiffractionDerivative, vec![F, NK, H], take(&mut out.kdt));
        }
    }
    if mode_matrices {
        put(&mut r, QuantityId::ModeRestoring, vec![D, D], Cp);
        put(&mut r, QuantityId::PressureModeDamping, vec![D, D], Bp);
    }
    if m.has_field_points {
        let Pa = m.field_points.len() as i64;
        let (Fi, Hi, Ki) = (F as i32, H as i32, K as i32);
        if m.opt.field_pressure {
            // elevation at z = 0, p / (rho g) below (first_order.field_pressure)
            // over all field points, those in tanks from their tank problems
            if pl.radiation {
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsPressureRadiation,
                    vec![F, Pa, K],
                    field_all(m, pl, take(&mut out.frad), Fi, Ki, true, false),
                );
            }
            if pl.diffraction {
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsPressureIncident,
                    vec![F, Pa, H],
                    field_all(m, pl, take(&mut out.finc), Fi, Hi, false, false),
                );
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsPressureScattering,
                    vec![F, Pa, H],
                    field_all(m, pl, take(&mut out.fsca), Fi, Hi, false, false),
                );
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsPressureDiffraction,
                    vec![F, Pa, H],
                    field_all(m, pl, take(&mut out.ftot), Fi, Hi, false, false),
                );
            }
        }
        if pl.fvel {
            if pl.radiation {
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsVelocityRadiation,
                    vec![F, Pa, K, 3],
                    field_all(m, pl, take(&mut out.fvr), Fi, 3 * Ki, true, true),
                );
            }
            if pl.diffraction {
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsVelocityIncident,
                    vec![F, Pa, H, 3],
                    field_all(m, pl, take(&mut out.fvi), Fi, 3 * Hi, false, true),
                );
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsVelocityScattering,
                    vec![F, Pa, H, 3],
                    field_all(m, pl, take(&mut out.fvs), Fi, 3 * Hi, false, true),
                );
                put_vec(
                    &mut r,
                    QuantityId::FieldPointsVelocityDiffraction,
                    vec![F, Pa, H, 3],
                    field_all(m, pl, take(&mut out.fvt), Fi, 3 * Hi, false, true),
                );
            }
        }
    }
    Ok(r)
}

/// The lid outputs: L points (global, [L][3]) with their one-based lid ids, and the elevations.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn put_lid_outputs(r: &mut Results, pl: &Plan, L: i32, pts: &[f64], ids: &[i32], out: &mut Output) {
    let F = pl.omegas.len() as i64;
    let H = pl.heads.len() as i64;
    let K = pl.rad.len() as i64;
    let L = L as i64;
    put(r, QuantityId::Lids, vec![L, 3], pts);
    put(r, QuantityId::LidsIds, vec![L], ids);
    if pl.radiation {
        put_vec(r, QuantityId::LidsElevationRadiation, vec![F, L, K], take(&mut out.lrad));
    }
    if pl.diffraction {
        put_vec(r, QuantityId::LidsElevationDiffraction, vec![F, L, H], take(&mut out.ldif));
    }
}
/// The pressure-surface outputs: Q points with their one-based surface numbers, and the elevations.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn put_pressure_outputs(r: &mut Results, pl: &Plan, Q: i32, pts: &[f64], ids: &[i32], out: &mut Output) {
    let F = pl.omegas.len() as i64;
    let H = pl.heads.len() as i64;
    let K = pl.rad.len() as i64;
    let Q = Q as i64;
    put(r, QuantityId::PressureSurfaces, vec![Q, 3], pts);
    put(r, QuantityId::PressureSurfacesIds, vec![Q], ids);
    if pl.radiation {
        put_vec(r, QuantityId::PressureSurfacesElevationRadiation, vec![F, Q, K], take(&mut out.prad));
    }
    if pl.diffraction {
        put_vec(r, QuantityId::PressureSurfacesElevationDiffraction, vec![F, Q, H], take(&mut out.pdif));
    }
}
/// The body outputs at Nb points: points, normals, areas, ids (body, element or point, reflection),
/// and the pressures and velocities [F][Nb][K or H] and [..][3] the plan asks for.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn put_body_outputs(
    r: &mut Results,
    pl: &Plan,
    Nb: i32,
    pts: &[f64],
    nrm: &[f64],
    area: &[f64],
    ids: &[i32],
    bpr: Vec<Complex32>,
    bpd: Vec<Complex32>,
    bvr: Vec<Complex32>,
    bvd: Vec<Complex32>,
) {
    let F = pl.omegas.len() as i64;
    let H = pl.heads.len() as i64;
    let K = pl.rad.len() as i64;
    let Nb = Nb as i64;
    put(r, QuantityId::BodyPoints, vec![Nb, 3], pts);
    put(r, QuantityId::BodyPointsNormals, vec![Nb, 3], nrm);
    put(r, QuantityId::BodyPointsAreas, vec![Nb], area);
    put(r, QuantityId::BodyPointsIds, vec![Nb, 4], ids);
    if pl.bpres && pl.radiation {
        put_vec(r, QuantityId::BodyPointsPressureRadiation, vec![F, Nb, K], bpr);
    }
    if pl.bpres && pl.diffraction {
        put_vec(r, QuantityId::BodyPointsPressureDiffraction, vec![F, Nb, H], bpd);
    }
    if pl.bvel && pl.radiation {
        put_vec(r, QuantityId::BodyPointsVelocityRadiation, vec![F, Nb, K, 3], bvr);
    }
    if pl.bvel && pl.diffraction {
        put_vec(r, QuantityId::BodyPointsVelocityDiffraction, vec![F, Nb, H, 3], bvd);
    }
}
