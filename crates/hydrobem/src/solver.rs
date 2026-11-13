//! The solve of a validated model (the route selected in route.rs, the motion matrices in
//! motion.rs): the hydrostatics route, the dispatch to the higher-order method (ho_solver.rs)
//! and the low-order routes here (infinite and finite depth, open sea, walls and a channel,
//! interior surfaces, damping lids, thin sheets, pressure surfaces, the omega = 0 / inf limits).
//!
//! Formulation (normal n out of the body into the fluid, time factor e^{iwt}):
//!   2 pi phi(x) - PV int phi dG/dn dS = - int G dphi/dn dS      on the body,
//!   radiation dphi_j/dn = n_j; diffraction: total 2 pi phi_D - PV int phi_D dG/dn = 4 pi phi_I,
//!   or scattered with dphi_S/dn = -dphi_I/dn;
//!   4 pi phi(x) = int phi dG/dn dS - int G dphi/dn dS            in the fluid.
//! Forces: A - iB/w = -rho int n_i phi_j dS,
//!   X = i w rho int n_i phi_D dS,  Haskind X_i = i w rho int (n_i phi_I - phi_i dphi_I/dn) dS.
//! Discretization: constant panels, centroid collocation, exact Rankine
//! integrals of the image terms, analytic integral of the logarithmic
//! singularity -2K psi, psi = log(r' + |z + zeta|) (theory §6.2), centroid
//! quadrature of the remainder. Interior surfaces, lids, walls
//! and symmetry sectors are described where they are built. The grounds of these choices (the
//! panel-rule tiers, the log split and its pair tolerance, the single-precision systems) are in
//! theory §6.1, §6.2 and §13.1.

use std::mem::take;
use std::sync::Arc;

use crate::codes::QuantityId;
use crate::constants::PI;
use crate::env_hooks::diag_double;
use crate::error::Result;
use crate::geometry::{Placement, Vec3, norm, v3};
use crate::green_fin::{FreqMode, verify_fin_tables};
use crate::green_inf::verify_green_tables;
use crate::ho_solver::solve_higher_order;
use crate::interior::InteriorPanels;
use crate::lo_frequency::solve_frequency;
use crate::lo_geometry::build_geometry;
use crate::lo_static::compute_static;
use crate::low_order::{Geometry, LazyRows, Output, Plan, Profile, Real, Static, SurfaceKind, field_statics_lazy};
use crate::model_codes::{NewmanDrifts, QTF_DIFFERENCES, QTF_SUMS};
use crate::parallel::pool;
use crate::problem::Problem;
use crate::progress::SolveContext;
use crate::results::{Results, put};
use crate::route::{Route, select_route};
use crate::solve_common::add_generalized_restoring;
use crate::solve_common::add_pressure_point;
use crate::solve_common::base_plan;
use crate::solve_common::channel_extent;
use crate::solve_common::{frequency_context, plan_coordinates};
use crate::solve_driver::{
    allocate_outputs, common_results, finish_plan, put_body_outputs, put_lid_outputs, put_pressure_outputs,
    run_frequencies,
};
use num_complex::Complex32;

/// Rigid-body motion matrices (theory §3.1), 6N x 6N row-major about each R.
#[derive(Clone, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct MotionMatrices {
    pub ready: bool, // every body has a centre of gravity and inertia
    pub M: Vec<f64>, // inertia
    pub C: Vec<f64>, // hydrostatic restoring with weight + additional stiffness
    pub B: Vec<f64>, // additional damping
}

/// The memory a solve of the model will use, in bytes. The parts are those of the model's own
/// solve; with the potential QTF the peak and the commit charge are the larger of the model's own
/// and those of its sub-problems, which run after it with its results held (memory.rs).
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryEstimate {
    pub statics: i64,       // shared by all frequencies: the influence data and the geometry
    pub per_frequency: i64, // one frequency thread: its matrices, LU and potentials (resident)
    pub outputs: i64,       // the outputs of all frequencies and the results built from them
    pub peak: i64,          // resident: statics + frequency_threads x per_frequency + outputs (or more, above)
    pub commit: i64,        // the commit charge: peak and the whole packing buffers of the threads
}

// The restoring of the pressure coordinates (theory §9.3)
// and of the generalized modes over the wetted and plate panels, D x D.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn panel_restoring(m: &Problem, g: &Geometry, Dn: i32) -> Vec<f64> {
    let mut Cp = vec![0.0f64; Dn as usize * Dn as usize];
    for j in 0..g.Nf as usize {
        if g.psurf[j] >= 0 && g.mirror[j] == 0 {
            add_pressure_point(m, &g.ps[g.psurf[j] as usize], &g.lc[j], g.pg[j].area, Dn, &mut Cp);
        }
    }
    for j in 0..g.Nf as usize {
        if (g.kind[j] == SurfaceKind::Wet || g.kind[j] == SurfaceKind::Plate)
            && g.mirror[j] == 0
            && g.body[j] >= 0
            && !m.bodies[g.body[j] as usize].generalized.is_empty()
        {
            let bi = g.body[j] as usize;
            let mut ng = Vec::new();
            let mut wg = Vec::new();
            let mut k0 = 0;
            for b in 0..bi {
                k0 += m.bodies[b].generalized.len();
            }
            for k in 0..m.bodies[bi].generalized.len() {
                ng.push(g.gn[k0 + k][j]);
                wg.push(g.gw[k0 + k][j]);
            }
            add_generalized_restoring(m, bi as i32, &g.nj[j], &ng, g.pg[j].area, &wg, Dn, &mut Cp);
        }
    }
    Cp
}

// The panels j with out_index[j] >= 0 at that index: centroids in global axes and their ids.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn frame_points(
    g: &Geometry,
    out_index: &[i32],
    n: i32,
    pts: &mut Vec<f64>,
    ids: &mut Vec<i32>,
    id: impl Fn(usize) -> i32,
) {
    *pts = vec![0.0; n as usize * 3];
    *ids = vec![0; n as usize];
    let frame = Placement { position: v3(g.px, g.py, 0.0), yaw_deg: g.yaw };
    for j in 0..g.Nf as usize {
        let q = out_index[j];
        if q < 0 {
            continue;
        }
        let q = q as usize;
        let c = frame.apply(g.pg[j].c);
        for k in 0..3 {
            pts[3 * q + k] = c[k];
        }
        ids[q] = id(j);
    }
}

// The body outputs at the body panels pl.bp: global centroids and normals, ids (body, element,
// reflection), areas.
fn put_body_panels(g: &Geometry, pl: &Plan, out: &mut Output, r: &mut Results) {
    let frame = Placement { position: v3(g.px, g.py, 0.0), yaw_deg: g.yaw };
    let mut pts = Vec::new();
    let mut nrm = Vec::new();
    let mut area = Vec::new();
    let mut bids: Vec<i32> = Vec::new();
    for &j in &pl.bp {
        let j = j as usize;
        let c = frame.apply(g.pg[j].c);
        let n = frame.rotate(g.pg[j].n);
        pts.extend_from_slice(&[c[0], c[1], c[2]]);
        nrm.extend_from_slice(&[n[0], n[1], n[2]]);
        area.push(g.pg[j].area);
        bids.extend_from_slice(&[g.body[j], g.rec[j] + 1, g.lref[j].sx as i32, g.lref[j].sy as i32]);
    }
    put_body_outputs(
        r,
        pl,
        pl.bp.len() as i32,
        &pts,
        &nrm,
        &area,
        &bids,
        take(&mut out.bpr),
        take(&mut out.bpd),
        take(&mut out.bvr),
        take(&mut out.bvd),
    );
}

// The body outputs at the specified body points: the
// inverse-distance mean of the body_point_neighbors nearest panel centroids of the same body
// (frame distances).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn put_specified_body_points(m: &Problem, g: &Geometry, pl: &Plan, out: &mut Output, r: &mut Results) {
    let F = pl.omegas.len();
    let H = pl.heads.len();
    let K = pl.rad.len();
    let Nb = pl.bp.len();
    let frame = Placement { position: v3(g.px, g.py, 0.0), yaw_deg: g.yaw };
    let mut pts = Vec::new();
    let mut nrm = Vec::new();
    let mut area = Vec::new();
    let mut bids: Vec<i32> = Vec::new();
    let mut wts: Vec<Vec<(usize, f64)>> = Vec::new(); // per point: (index into pl.bp, weight)
    for (bi, b) in m.bodies.iter().enumerate() {
        let Nn = b.body_point_neighbors;
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        for (i, bp) in b.body_points.iter().enumerate() {
            let xg = place.apply(*bp);
            // into the frame: x_frame = R(-yaw) (x_global - (px, py, 0))
            let cy = (g.yaw * PI / 180.0).cos();
            let sy = (g.yaw * PI / 180.0).sin();
            let dx = xg[0] - g.px;
            let dy = xg[1] - g.py;
            let xf = v3(cy * dx + sy * dy, -sy * dx + cy * dy, xg[2]);
            let mut near: Vec<(f64, usize)> = Vec::new();
            for q in 0..pl.bp.len() {
                if g.body[pl.bp[q] as usize] == bi as i32 {
                    near.push((norm(g.pg[pl.bp[q] as usize].c - xf), q));
                }
            }
            let n = near.len().min(Nn as usize);
            // a full stable sort of the pairs in (distance, index) order; the first n are the nearest
            near.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut w = Vec::new();
            if near[0].0 <= 1e-12 * 1.0f64.max(norm(xf)) {
                w.push((near[0].1, 1.0));
            } else {
                let mut sum = 0.0;
                for x in &near[..n] {
                    sum += 1.0 / x.0;
                }
                for x in &near[..n] {
                    w.push((x.1, 1.0 / x.0 / sum));
                }
            }
            wts.push(w);
            let nn = frame.rotate(g.pg[pl.bp[near[0].1] as usize].n);
            pts.extend_from_slice(&[xg[0], xg[1], xg[2]]);
            nrm.extend_from_slice(&[nn[0], nn[1], nn[2]]);
            area.push(0.0);
            bids.extend_from_slice(&[bi as i32, i as i32 + 1, 0, 0]);
        }
    }
    let Np = wts.len();
    let interp = |inp: Vec<Complex32>, per: usize| -> Vec<Complex32> {
        // [F][Nb][per] -> [F][Np][per]
        let mut o = vec![Complex32::new(0.0, 0.0); if inp.is_empty() { 0 } else { F * Np * per }];
        if inp.is_empty() {
            return o;
        }
        for f in 0..F {
            for q in 0..Np {
                for &(j, wj) in &wts[q] {
                    for c in 0..per {
                        o[(f * Np + q) * per + c] += (wj as f32) * inp[(f * Nb + j) * per + c];
                    }
                }
            }
        }
        o
    };
    put_body_outputs(
        r,
        pl,
        Np as i32,
        &pts,
        &nrm,
        &area,
        &bids,
        interp(take(&mut out.bpr), K),
        interp(take(&mut out.bpd), H),
        interp(take(&mut out.bvr), 3 * K),
        interp(take(&mut out.bvd), 3 * H),
    );
}

/// The static rows of the body panels, of the field points (shared, Problem::field_statics) and of
/// the control-surface points.
type Statics<R> = (Static<R>, Arc<Static<R>>, Static<R>);

// The influence data and the frequencies in the real type R: float32, or double in the diagnostic
// mode. The statics on frequency_threads x inner_threads threads, then the frequencies
// (run_frequencies), each into outputs of its own (frequency_output) written into `out`.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn run<R: Real>(
    m: &Problem,
    g: &Geometry,
    pl: &Plan,
    cents: &[Vec3],
    depth: f64,
    want_inf: bool,
    xext: f64,
    prof: &mut Profile,
    ctx: &SolveContext,
    frequency_output: &(dyn Fn() -> Output + Sync),
    out: &mut Output,
) -> Result<()> {
    let work = ctx.statics();
    let statics = pool("statics", m.opt.total_threads())?;
    let (st, sf, scs) = statics.install(|| -> Result<Statics<R>> {
        let st: Static<R> =
            compute_static(g, cents, pl.logint, m.opt.log_tolerance, depth, want_inf, true, false, 0, work)?;
        prof.mark("static (body)");
        // the velocity rows of the field points for their velocity only, those of the control-surface
        // points always
        // (shared with the next solve of the same geometry and points when the problem asks)
        let cached =
            m.field_statics.as_ref().and_then(|c| c.0.get()).and_then(|a| a.clone().downcast::<Static<R>>().ok());
        let sf: Arc<Static<R>> = match cached {
            Some(s) => s,
            None => {
                let vfrom = m.opt.field_velocity_from.min(g.fp.len());
                let vrows = if g.source && m.opt.field_velocity { g.fp.len() - vfrom } else { 0 };
                let lazy = field_statics_lazy(g.fp.len(), g.Nf as usize, vrows, std::mem::size_of::<R>(), g.W > 0.0);
                let s = Arc::new(if lazy {
                    // too large to store: each point's rows when it is evaluated
                    Static::<R> {
                        vfrom,
                        lazy: Some(LazyRows { logint: pl.logint, tol: m.opt.log_tolerance, depth }),
                        ..Default::default()
                    }
                } else {
                    compute_static(
                        g,
                        &g.fp,
                        pl.logint,
                        m.opt.log_tolerance,
                        depth,
                        false,
                        false,
                        m.opt.field_velocity,
                        m.opt.field_velocity_from,
                        work,
                    )?
                });
                if let Some(c) = &m.field_statics {
                    let _ = c.0.set(s.clone() as Arc<dyn std::any::Any + Send + Sync>);
                }
                s
            }
        };
        prof.mark("static (field points)");
        let csx: Vec<Vec3> = g.cs_reps.iter().map(|&q| g.cs[q as usize].x).collect();
        let scs: Static<R> = compute_static(
            g,
            &csx,
            pl.logint,
            m.opt.log_tolerance,
            depth,
            false,
            false,
            m.opt.control_surface_drift,
            0,
            work,
        )?;
        Ok((st, sf, scs))
    })?;
    drop(statics);
    run_frequencies(m, pl, ctx, out, &|mode: FreqMode, fi: i32, work| {
        let fc = frequency_context(m, g.W, pl, mode, if fi < 0 { 0.0 } else { pl.omegas[fi as usize] }, xext)?;
        let mut o = frequency_output();
        solve_frequency(m, g, pl, &st, &sf, &fc, fi, work, &mut o, if fi < 0 { None } else { Some(&scs) })?;
        Ok(o)
    })
}

/// Solve a validated model on its route (low order here, higher order in solve_higher_order()).
/// m: problem; ctx: the cancel request and the progress reports of the solve (a tank
/// sub-problem follows its model's cancel request). Returns the results, keyed by QuantityId.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn solve(m: &mut Problem, ctx: &SolveContext) -> Result<Box<Results>> {
    let mut r = solve_route(m, ctx)?;
    if m.opt.potential_qtf.any() {
        // the force of the second-order potential: its sub-problems after the model's solve, the
        // methods asked for sharing them (each under its own quantities)
        crate::potential_qtf::add_potential_qtfs(m, ctx, &mut r)?;
    }
    // Newman's approximation from the drifts of the frequencies (that from the momentum drift
    // follows it in SolveJob::run)
    newman(m, &mut r, NewmanDrifts { pressure_integration: true, control_surface: true, momentum_conservation: false });
    Ok(r)
}

/// The drifts and QTFs as the model asked for them (at the end of SolveJob::run): without the
/// quantities computed only for others (Options::hidden: the pressure drift and QTFs the quadratic
/// parts rest on, the QTFs of the bidirectional drifts, the frequencies not asked for), and
/// without the heading pairs of none.
pub(crate) fn drop_hidden(m: &Problem, r: &mut Results) {
    use QuantityId as Q;
    for q in m.opt.hidden.iter() {
        r.quantities.remove(&q);
    }
    // the heading pairs of what is left
    if !QTF_SUMS.iter().chain(&QTF_DIFFERENCES).any(|q| r.quantities.contains_key(q)) {
        r.quantities.remove(&Q::QtfHeadingPairs);
    }
    let drifts = [Q::DriftPressureIntegration, Q::DriftControlSurface, Q::DriftMomentumConservation];
    if !drifts.iter().any(|q| r.quantities.contains_key(q)) {
        r.quantities.remove(&Q::DriftHeadingPairs);
    }
}

/// Newman's approximation of the difference-frequency QTF (theory §12.13; Newman 1974) from the
/// mean drifts of the equal heading pairs: T_ij = sign(T_ii) sqrt(T_ii T_jj) where the two drifts
/// have the same sign, 0 where not (geometric), or (T_ii + T_jj)/2 (arithmetic); the drift on the
/// diagonal; NaN where a drift is NaN (a coordinate without it). [F, F, H, n], real (the mean drift
/// of equal headings is real). `which`: the methods whose drift is there now.
pub(crate) fn newman(m: &Problem, r: &mut Results, which: NewmanDrifts) {
    use QuantityId as Q;
    let o = &m.opt;
    let methods = [
        (
            o.newman.pressure_integration && which.pressure_integration,
            Q::DriftPressureIntegration,
            Q::QtfNewmanPressureIntegrationDifference,
        ),
        (
            o.newman.control_surface && which.control_surface,
            Q::DriftControlSurface,
            Q::QtfNewmanControlSurfaceDifference,
        ),
        (
            o.newman.momentum_conservation && which.momentum_conservation,
            Q::DriftMomentumConservation,
            Q::QtfNewmanMomentumConservationDifference,
        ),
    ];
    let pairs: Vec<f64> =
        r.quantities.get(&Q::DriftHeadingPairs).map(|v| crate::results::elements_of(v.data())).unwrap_or_default();
    let np = pairs.len() / 2;
    let heads = &m.headings;
    // the pair of each heading with itself
    let at: Vec<Option<usize>> = heads
        .iter()
        .map(|&h| (0..np).find(|&p| (pairs[2 * p] - h).abs() <= 1e-9 && (pairs[2 * p + 1] - h).abs() <= 1e-9))
        .collect();
    for (on, qd, qn) in methods {
        if !on {
            continue;
        }
        let Some(dq) = r.quantities.get(&qd) else { continue };
        let v: Vec<f64> = crate::results::elements_of(dq.data()); // complex: (re, im)
        let (nf, n) = (dq.dims[0] as usize, dq.dims[2] as usize);
        let nh = heads.len();
        let drift = |f: usize, p: usize, c: usize| v[2 * ((f * np + p) * n + c)];
        let mut t = vec![0.0f64; nf * nf * nh * n];
        for (h, ph) in at.iter().enumerate() {
            let Some(p) = *ph else { continue };
            for i in 0..nf {
                for j in 0..nf {
                    for c in 0..n {
                        let (a, b) = (drift(i, p, c), drift(j, p, c));
                        let x = if a.is_nan() || b.is_nan() {
                            f64::NAN // a coordinate without the drift (reference/results)
                        } else if o.newman_geometric {
                            if a * b > 0.0 { a.signum() * (a * b).sqrt() } else { 0.0 }
                        } else {
                            0.5 * (a + b)
                        };
                        t[((i * nf + j) * nh + h) * n + c] = if i == j { a } else { x };
                    }
                }
            }
        }
        crate::results::put_vec(r, qn, vec![nf as i64, nf as i64, nh as i64, n as i64], t);
    }
}

// the solve of the route
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn solve_route(m: &mut Problem, ctx: &SolveContext) -> Result<Box<Results>> {
    let mut frequencies = 0; // the ordinary frequencies and the limits (radiation only), as run_frequencies
    for &w in &m.omegas {
        // an ordinary frequency, or a limit (radiation only)
        frequencies += if (w > 0.0 && w.is_finite()) || m.opt.radiation { 1 } else { 0 };
    }
    ctx.reporter.set_frequencies(frequencies);
    let route = select_route(m)?;
    if route == Route::Hydrostatics {
        // no boundary-value problem: the hydrostatics are the model query Model::hydrostatics
        let mut r = Box::new(Results { route: route.name(), ..Default::default() });
        let none: Vec<f64> = Vec::new();
        put(&mut r, QuantityId::Omegas, vec![0], &none);
        put(&mut r, QuantityId::Headings, vec![m.headings.len() as i64], &m.headings);
        return Ok(r);
    }
    if m.opt.higher_order {
        verify_green_tables()?;
        if m.env.depth.is_finite() {
            verify_fin_tables()?;
        }
        return solve_higher_order(m, &route, ctx);
    }
    let m: &Problem = m;
    verify_green_tables()?;
    let depth = m.env.depth;
    if depth.is_finite() {
        verify_fin_tables()?;
    }
    let mut pl = base_plan(m);

    let mut prof = Profile::new();
    let mut ints: Vec<InteriorPanels> = Vec::new();
    let mut g = build_geometry(m, &mut ints)?;
    g.source = m.opt.pressure_drift || m.opt.field_velocity || m.opt.body_velocity || m.opt.control_surface_drift;
    prof.mark("geometry");
    plan_coordinates(m, &g.ps, &mut pl);
    let cents: Vec<Vec3> = (0..g.Ns as usize).map(|k| g.pg[g.sec[k] as usize].c).collect();
    let want_inf = m.has_infinite() && pl.radiation;
    let mut Cp = panel_restoring(m, &g, pl.D);
    let mut Bp = vec![0.0f64; Cp.len()];
    finish_plan(m, &g.ps, &mut Cp, &mut Bp, &mut pl, ctx)?;

    if pl.bpres || pl.bvel {
        for j in 0..g.Nf as usize {
            let kd = g.kind[j];
            if (kd == SurfaceKind::Wet || kd == SurfaceKind::Press || kd == SurfaceKind::Plate)
                && g.mirror[j] == 0
                && g.body[j] >= 0
            {
                pl.bp.push(j as i32); // the .5 files of the format list the pressure-surface panels too
            }
        }
    }
    let Nb = pl.bp.len() as i32;
    let P = g.fp.len() as i32;
    let mut out = allocate_outputs(m, &pl, pl.omegas.len(), P, g.L, g.Q, Nb);
    let frequency_output = || allocate_outputs(m, &pl, 1, P, g.L, g.Q, Nb);
    // channel: the extent in x of all panel centroids, field and control-surface points (tail tiles)
    let xext = channel_extent(&g, &g.fp);

    // influence data and matrices in float32, or double in the diagnostic mode
    if diag_double() {
        run::<f64>(m, &g, &pl, &cents, depth, want_inf, xext, &mut prof, ctx, &frequency_output, &mut out)?;
    } else {
        run::<f32>(m, &g, &pl, &cents, depth, want_inf, xext, &mut prof, ctx, &frequency_output, &mut out)?;
    }
    prof.mark("frequencies");

    let mut r = common_results(m, &pl, &route, &mut out, &Cp, &Bp, g.Q > 0 || !g.gn.is_empty(), ctx)?;
    if g.L > 0 {
        let mut pts = Vec::new();
        let mut ids = Vec::new();
        frame_points(&g, &g.lid_out, g.L, &mut pts, &mut ids, |j| g.lid[j] + 1);
        put_lid_outputs(&mut r, &pl, g.L, &pts, &ids, &mut out);
    }
    if !g.ps.is_empty() {
        let mut pts = Vec::new();
        let mut ids = Vec::new();
        frame_points(&g, &g.press_out, g.Q, &mut pts, &mut ids, |j| g.psurf[j] + 1);
        put_pressure_outputs(&mut r, &pl, g.Q, &pts, &ids, &mut out);
    }
    if (pl.bpres || pl.bvel) && !m.bodies.is_empty() && !m.bodies[0].body_points.is_empty() {
        put_specified_body_points(m, &g, &pl, &mut out, &mut r);
    } else if pl.bpres || pl.bvel {
        put_body_panels(&g, &pl, &mut out, &mut r);
    }
    Ok(r)
}
