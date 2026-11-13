//! Higher-order method (theory §7; Maniar 1995).
//!
//! The potential on each patch is a tensor product of B-splines of orders
//! (k_u, k_v) on n_u x n_v uniform panels of the parametric square [-1, 1]^2; the
//! boundary integral equation (normals into the fluid)
//!   2 pi phi(x) - int phi dG/dn dS = -int G u dS
//! is tested with the basis functions in parametric space (Galerkin): outer Gauss
//! points of the outer quadrature order with parametric weights, inner Gauss points
//! of the inner quadrature order with the area element. The Rankine parts
//! (with the free-surface and bottom images) and the logarithmic singularity
//! of the wave kernel are integrated by tiers: a 3 x 3 rule beyond 12 block sizes, the
//! inner rule beyond 6, a 6 x 6 rule beyond 1.5, adaptive subdivision nearer and four
//! graded Duffy-transformed rectangles on the panel containing the target (graded in
//! proportion to the aspect of each triangle, so that elongated panels keep their accuracy),
//! the images by their own distances; the regular wave part uses the inner rule, and a rule
//! of one order below it beyond 3 sizes from the images. Symmetry sectors and characters as
//! in the low-order method: a single body in open sea in its horizontal frame
//! with the mesh reflections, else the global frame with the reflections of the
//! whole system and wall images.

use crate::codes::Stage;
use std::mem::take;
use std::sync::atomic::{AtomicI32, Ordering};

use rayon::prelude::*;

use crate::error::Result;
use crate::geometry::{Placement, v3};
use crate::green_fin::FreqMode;
use crate::ho_frequency::solve_frequency;
use crate::ho_geometry::build;
use crate::ho_geometry::build_blocks;
use crate::ho_geometry::{build_projections, build_wave_arrays};
use crate::ho_internal::{HoGeom, MAX_MODES, SheetEval, StaticRow};
use crate::ho_sheets::sheet_rankine_row;
use crate::ho_static::static_row;
use crate::low_order::{Output, Plan, Profile, SurfaceKind};
use crate::parallel::pool;
use crate::problem::Problem;
use crate::progress::{SolveContext, Work};
use crate::results::Results;
use crate::route::Route;
use crate::solve_common::add_generalized_restoring;
use crate::solve_common::add_pressure_point;
use crate::solve_common::base_plan;
use crate::solve_common::frequency_context;
use crate::solve_common::plan_coordinates;
use crate::solve_driver::{
    allocate_outputs, common_results, finish_plan, put_body_outputs, put_lid_outputs, put_pressure_outputs,
    run_frequencies,
};

// The restoring of the pressure coordinates (theory §9.3, as the low-order method) and of
// the generalized modes over the wetted patches, D x D.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn patch_restoring(m: &Problem, h: &HoGeom, Dn: i32) -> Vec<f64> {
    let mut Cp = vec![0.0f64; Dn as usize * Dn as usize];
    for q in &h.inner {
        if h.kind_of(q.inst) == SurfaceKind::Press && !h.inst[q.inst as usize].image {
            add_pressure_point(
                m,
                &h.ps[h.pss[h.inst[q.inst as usize].patch as usize] as usize],
                &v3(q.xl[0], q.xl[1], 0.0),
                q.w,
                Dn,
                &mut Cp,
            );
        }
    }
    for q in &h.inner {
        let b = h.inst[q.inst as usize].body;
        if (h.kind_of(q.inst) != SurfaceKind::Wet && h.kind_of(q.inst) != SurfaceKind::Plate)
            || h.inst[q.inst as usize].image
            || m.bodies[b as usize].generalized.is_empty()
        {
            continue;
        }
        let mut ng = Vec::new();
        let mut wg = Vec::new();
        for k in 0..h.gmode.len() {
            if h.gm_body[k] == b {
                ng.push(q.nm[6 * h.NB as usize + k]);
                wg.push(h.gmode[k].displacement(&q.xl, h.prec[h.inst[q.inst as usize].patch as usize])[2]);
            }
        }
        let nr: &[f64; 6] = q.nm[6 * b as usize..6 * b as usize + 6].try_into().unwrap();
        add_generalized_restoring(m, b, nr, &ng, q.w, &wg, Dn, &mut Cp);
    }
    Cp
}

// The rows of the planar dipole patches: the outer points of the sheet instances with the
// test functions of their panel, and the Rankine part of their integrals; with omega = inf also
// those of the free-surface plates, sheets of the odd-image problem at omega = inf only (theory
// §9.8). On the threads of the current pool, each evaluation point its own rows.
fn sheet_rows(h: &HoGeom, want_inf: bool, work: Work<'_>) -> Result<Vec<SheetEval>> {
    let mut se = Vec::new();
    for o in &h.outer {
        let kd = h.kind_of(o.inst);
        if kd != SurfaceKind::Sheet && !(want_inf && kd == SurfaceKind::Plate) {
            continue;
        }
        let dc = h.disc_of(o.inst);
        let mut ev = SheetEval::default();
        for b in 0..dc.kv {
            for a in 0..dc.ku {
                ev.rw.push((
                    h.dof0[o.inst as usize] + (o.iv + b) * dc.mu() + o.iu + a,
                    o.w * o.bu[a as usize] * o.bv[b as usize],
                ));
            }
        }
        ev.w = o.w;
        ev.sp.inst = o.inst;
        ev.sp.u = o.uf;
        ev.sp.v = o.vf;
        ev.sp.iu = o.iu;
        ev.sp.iv = o.iv;
        ev.sp.x = o.x;
        ev.sp.n = o.n;
        ev.sp.nm[..MAX_MODES].copy_from_slice(&o.nm[..MAX_MODES]);
        ev.plate = kd == SurfaceKind::Plate;
        se.push(ev);
    }
    se.par_iter_mut().try_for_each(|e| -> Result<()> {
        work.check_cancel()?;
        if !e.plate {
            sheet_rankine_row(h, h.depth, false, &e.sp, &mut e.reg);
        }
        if want_inf {
            sheet_rankine_row(h, h.depth, true, &e.sp, &mut e.inf);
        }
        Ok(())
    })?;
    Ok(se)
}

// The inner points qi with out_index[qi] >= 0 at that index: global points and their ids.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn frame_points(
    h: &HoGeom,
    out_index: &[i32],
    n: i32,
    pts: &mut Vec<f64>,
    ids: &mut Vec<i32>,
    id: impl Fn(usize) -> i32,
) {
    *pts = vec![0.0; n as usize * 3];
    *ids = vec![0; n as usize];
    let frame = Placement { position: v3(h.px, h.py, 0.0), yaw_deg: h.yaw };
    for qi in 0..h.inner.len() {
        let q = out_index[qi];
        if q < 0 {
            continue;
        }
        let q = q as usize;
        let c = frame.apply(h.inner[qi].x);
        for k in 0..3 {
            pts[3 * q + k] = c[k];
        }
        ids[q] = id(qi);
    }
}

// The body output points: global points and normals, ids (body, point, reflection), J du dv.
fn put_body_points(h: &HoGeom, pl: &Plan, out: &mut Output, r: &mut Results) {
    let frame = Placement { position: v3(h.px, h.py, 0.0), yaw_deg: h.yaw };
    let mut pts = Vec::new();
    let mut nrm = Vec::new();
    let mut area = Vec::new();
    let mut bids: Vec<i32> = Vec::new();
    for bp in &h.bpts {
        let c = frame.apply(bp.x);
        let n = frame.rotate(bp.n);
        pts.extend_from_slice(&[c[0], c[1], c[2]]);
        nrm.extend_from_slice(&[n[0], n[1], n[2]]);
        area.push(bp.area);
        let in_ = &h.inst[bp.inst as usize];
        let sx = if bp.given { 0 } else { in_.r.sx as i32 };
        let sy = if bp.given { 0 } else { in_.r.sy as i32 };
        bids.extend_from_slice(&[bp.body, bp.rec, sx, sy]);
    }
    put_body_outputs(
        r,
        pl,
        h.bpts.len() as i32,
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

/// A channel: the extent in x of every point of the wave rows (the targets and the source points
/// of the tail tiles), else 0.
fn channel_extent(h: &HoGeom) -> f64 {
    if h.W <= 0.0 {
        return 0.0;
    }
    let xs = h.wx.iter().step_by(3).chain(h.fx.iter().step_by(3));
    let xs = xs.chain(h.outer.iter().map(|p| &p.x[0])).chain(h.fp.iter().map(|p| &p[0]));
    let (lo, hi) = xs.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    hi - lo
}

/// Solve a problem with the higher-order method.
/// m: problem; route: its route from select_route(); ctx: the cancel request and the progress
/// reports of the solve. Returns the results, keyed by QuantityId.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn solve_higher_order(m: &Problem, route: &Route, ctx: &SolveContext) -> Result<Box<Results>> {
    let mut pl = base_plan(m);
    let mut prof = Profile::new();
    let mut hg = build(m)?;
    build_blocks(&mut hg);
    build_wave_arrays(&mut hg)?;
    build_projections(&mut hg, pl.diffraction && !pl.total && !pl.heads.is_empty())?;
    let h = &hg;
    prof.mark("ho geometry");
    plan_coordinates(m, &h.ps, &mut pl);
    let mut Cp = patch_restoring(m, h, pl.D);
    let mut Bp = vec![0.0f64; Cp.len()];
    let want_inf = m.has_infinite() && pl.radiation;
    let mut so: Vec<StaticRow> = vec![StaticRow::default(); h.outer.len()];
    let mut sf: Vec<StaticRow> = vec![StaticRow::default(); h.fp.len()];
    let nrows = (h.outer.len() + h.fp.len()) as i32;
    let done = AtomicI32::new(0); // the progress of the static rows
    // the static rows on frequency_threads x inner_threads threads, each item its own row
    let work = ctx.statics();
    let row_done = || work.report(Stage::Statics, done.fetch_add(1, Ordering::SeqCst) + 1, nrows);
    let statics = pool("statics", m.opt.total_threads())?;
    let se = statics.install(|| -> Result<Vec<SheetEval>> {
        so.par_iter_mut().enumerate().try_for_each(|(f, row)| -> Result<()> {
            work.check_cancel()?;
            let o = &h.outer[f];
            // the outer points of sheets take collocation rows (sheet_rows), not these
            if h.kind_of(o.inst) != SurfaceKind::Sheet {
                static_row(h, &o.x, o.inst, o.uf, o.vf, pl.logint, want_inf, row);
            }
            row_done();
            Ok(())
        })?;
        sf.par_iter_mut().enumerate().try_for_each(|(q, row)| -> Result<()> {
            work.check_cancel()?;
            static_row(h, &h.fp[q], -1, 0.0, 0.0, pl.logint, false, row);
            row_done();
            Ok(())
        })?;
        sheet_rows(h, want_inf, work)
    })?;
    drop(statics);
    prof.mark("ho static");
    finish_plan(m, &h.ps, &mut Cp, &mut Bp, &mut pl, ctx)?;
    pl.fvel = h.fstep > 0.0;
    let nb = h.bpts.len() as i32;
    let xext = channel_extent(h);
    let mut out = allocate_outputs(m, &pl, pl.omegas.len(), h.Pu, h.L, h.Q, nb);
    {
        let pl = &pl;
        run_frequencies(m, pl, ctx, &mut out, &|mode: FreqMode, fi: i32, work| {
            let fc = frequency_context(m, h.W, pl, mode, if fi < 0 { 0.0 } else { pl.omegas[fi as usize] }, xext)?;
            // every frequency its own outputs, written into those of all after (run_frequencies)
            let mut o = allocate_outputs(m, pl, 1, h.Pu, h.L, h.Q, nb);
            solve_frequency(m, h, pl, &se, &so, &sf, &fc, fi, work, &mut o)?;
            Ok(o)
        })?;
    }
    prof.mark("ho frequencies");

    let mut r = common_results(m, &pl, route, &mut out, &Cp, &Bp, h.Q > 0 || !h.gmode.is_empty(), ctx)?;
    if h.L > 0 {
        // lid points: the inner points of the lid patches (global), one-based lid ids
        let mut pts = Vec::new();
        let mut ids = Vec::new();
        frame_points(h, &h.lid_out, h.L, &mut pts, &mut ids, |qi| h.lid_of(h.inner[qi].inst) + 1);
        put_lid_outputs(&mut r, &pl, h.L, &pts, &ids, &mut out);
    }
    if !h.ps.is_empty() {
        // pressure-surface points: the inner points of the pressure patches (global), one-based surfaces
        let mut pts = Vec::new();
        let mut ids = Vec::new();
        frame_points(h, &h.press_out, h.Q, &mut pts, &mut ids, |qi| {
            h.pss[h.inst[h.inner[qi].inst as usize].patch as usize] + 1
        });
        put_pressure_outputs(&mut r, &pl, h.Q, &pts, &ids, &mut out);
    }
    if pl.bpres || pl.bvel {
        put_body_points(h, &pl, &mut out, &mut r);
    }
    Ok(r)
}
