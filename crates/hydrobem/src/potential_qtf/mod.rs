//! The force of the second-order potential (theory §12.1; theory §12.4-§12.15), by the indirect
//! method (Molin 1979; Lee 1995, §3, §4.2) or the direct one (theory §12.6), for fixed and
//! floating bodies. For the waves a, b and Ω = ω_a ∓ ω_b, with ψ_j the radiation potential of mode
//! j at Ω (unit velocity),
//!   F_j = iρΩ [∫_B φ_I⁽²⁾ n_j dS + ∫_B ψ_j Q_B dS − (1/g) ∫_F ψ_j F_S dS],
//! φ_I⁽²⁾ the second-order incident wave, F_S the free-surface forcing of the disturbances
//! (scattering and the radiation of the first-order motions) and Q_B = −∂φ_I⁽²⁾/∂n plus, for a
//! moving body, the terms of its first-order motions (Lee 1995, eqs 3.11-3.15), whose second
//! derivatives are moved to first ones and a waterline integral by Stokes's theorem. The fields
//! come from two sub-problems the engine solves (§12.10: the diffraction, and for a moving body the
//! radiation and the motions, at the model's frequencies with the free-surface points as field
//! points; the radiation of the rigid and generalized modes at the distinct Ω); the free surface
//! inside the inner radius R by the panelled zone (§12.8: the generated panels, the cells or a given
//! mesh), beyond it from the fields on circles: across a quadrature zone the outgoing wave plus a
//! remainder interpolated in 1/r, beyond the outer circle R_OC the outgoing Hankel series of the
//! disturbances with a local (non-propagating) term per order, fitted on the circles r = R_OC and
//! r = 1.1 R_OC (§12.9). With one wall, the open-sea problem of the bodies and their mirror images,
//! its free surface integrated over the whole plane and halved (§12.14). A body with generalized
//! modes (theory §10.1): their radiation at Ω gives their generalized forces, their motions join
//! the disturbance and the body condition (their displacement, the turned normal of their
//! displacement gradient and its divergence in the transform of the second derivatives, §12.15).
//!
//! The modules: `bessel` (the Bessel functions), `far_field` (the fits beyond the panelled zone),
//! `forcing` (the free-surface forcing, the divergence theorem, the incident waves, the wall),
//! `body` (the body points, the waterline values, the motion terms of the body condition),
//! `zones` (the free surface's zones), `outer` (the integral beyond the outer circle), `items`
//! (the inner free surface and the body of every pair, both methods); here the field points, the
//! sub-problems and the order of the phases. Both methods asked for share everything but the
//! radiation at Ω and their own sums. Every sum runs in a fixed order, so that the results do not
//! depend on the thread counts.

mod bessel;
mod body;
mod far_field;
mod forcing;
mod items;
mod outer;
mod zones;

pub(crate) use zones::{Zones, zones};

use num_complex::Complex64;
use rayon::prelude::*;

use crate::codes::QuantityId;
use crate::constants::PI;
use crate::error::{Error, Result};
use crate::geometry::{Vec3, v3};
use crate::low_order::Profile;
use crate::model_codes::{NewmanDrifts, QtfHeadings, QuantitySet, WaterlineTerm};
use crate::problem::{Problem, SharedStatics};
use crate::progress::{SecondOrderStep, SolveContext};
use crate::results::{Results, elements_of, put_vec};
use crate::route::select_route;
use crate::solver::solve;
use body::{Layout, Motion, WlFs, free_surface_waterline};
use far_field::{Circles, Far, Zone, ZoneRadial, on_circle, ring_dft};
use forcing::{Wave, mirror_heading};

const CZ: Complex64 = Complex64::new(0.0, 0.0);
const I: Complex64 = Complex64::new(0.0, 1.0);
const CD: i64 = 16; // the bytes of a complex64 (the memory estimate)

fn read_c32(r: &Results, q: QuantityId) -> Vec<Complex64> {
    let v: Vec<f32> = r.quantities.get(&q).map(|v| elements_of(v.data())).unwrap_or_default();
    v.chunks(2).map(|c| Complex64::new(c[0] as f64, c[1] as f64)).collect()
}
fn read_f64(r: &Results, q: QuantityId) -> Vec<f64> {
    r.quantities.get(&q).map(|v| elements_of(v.data())).unwrap_or_default()
}
fn read_i32(r: &Results, q: QuantityId) -> Vec<i32> {
    r.quantities.get(&q).map(|v| elements_of(v.data())).unwrap_or_default()
}

// ---------------------------------------------------------------- the sub-problems

/// The model without any output it asked for: the base of a sub-problem, which asks for its own.
/// The second-order outputs, the field and body outputs, the user's field points and body points,
/// the control surfaces and the free-surface mesh are gone (the sub-problems take the body
/// panels as body points and the free-surface points as field points).
fn quiet(m: &Problem) -> Problem {
    let mut s = m.clone();
    let o = &mut s.opt;
    o.rao = false;
    o.drift_angles = 0;
    o.pressure_drift = false;
    o.pressure_qtf = QtfHeadings::None;
    o.control_surface_drift = false;
    o.control_surface_qtf = false;
    o.potential_qtf = QtfHeadings::None;
    o.potential_qtf_direct = false;
    o.potential_qtf_both = false;
    o.hidden = QuantitySet::default();
    o.newman = NewmanDrifts::default();
    o.body_pressure = false;
    o.body_velocity = false;
    o.field_pressure = false;
    o.field_velocity = false;
    o.field_velocity_from = 0;
    o.free_surface_mesh = None;
    o.potential_qtf_radius = None;
    o.potential_qtf_panel_size = None;
    s.field_points.clear();
    s.field_tanks.clear();
    s.has_field_points = false;
    s.field_statics = None;
    s.direct = None;
    for b in &mut s.bodies {
        b.body_points.clear();
        b.control_surface = None;
        b.control_surface_alternative = WaterlineTerm::ControlSurface;
    }
    s
}

/// The model's threads (frequency_threads x inner_threads) for a sub-problem of n frequencies: the
/// model's frequency workers (no more than its frequencies), so that a worker number of its
/// progress is one of the model's, each with the threads left inside its frequency (all threads
/// busy when the sub-problem has fewer frequencies than the model has workers). The results do not
/// depend on the split.
fn rebalance_threads(s: &mut Problem, n: usize) {
    let total = s.opt.total_threads().max(1);
    let workers = (n.max(1) as i32).min(s.opt.frequency_threads.max(1));
    s.opt.frequency_threads = workers;
    s.opt.inner_threads = (total / workers).max(1);
}

/// A sub-problem checked: the model rules without its field points (the free-surface points the
/// engine places, not points of the user to classify), then its route with them at `pts`. The model
/// passed its own rules, so a refusal of the sub-problem (its rules or its route) is an internal
/// error, not an error of the user's model.
fn checked(mut s: Problem, pts: Vec<Vec3>) -> Result<Problem> {
    let (fp, fv) = (s.opt.field_pressure, s.opt.field_velocity);
    s.opt.field_pressure = false;
    s.opt.field_velocity = false;
    s.invalidate();
    s.validate().map_err(|e| Error::internal(format!("potential QTF: a sub-problem is refused: {}", e.message())))?;
    s.warnings.clear();
    s.opt.field_pressure = fp;
    s.opt.field_velocity = fv;
    s.field_points = pts;
    s.has_field_points = true;
    select_route(&s)
        .map_err(|e| Error::internal(format!("potential QTF: a sub-problem has no route: {}", e.message())))?;
    Ok(s)
}

/// The first-order sub-problem (theory §12.10): the diffraction at the model's frequencies with
/// the body pressures and the field points, for a moving body also the radiation, the motions and
/// the body velocities (∇φ on the body in the motion terms of the body condition).
fn first_order_problem(
    m: &Problem,
    fp: &FieldPoints,
    omegas: &[f64],
    moving: bool,
    st: &SharedStatics,
) -> Result<Problem> {
    let mut sa = quiet(m);
    sa.field_statics = Some(st.clone());
    sa.omegas = omegas.to_vec();
    rebalance_threads(&mut sa, omegas.len());
    sa.opt.radiation = moving;
    sa.opt.rao = moving;
    sa.opt.body_velocity = moving;
    sa.opt.body_pressure = true;
    sa.opt.diffraction = true;
    sa.opt.field_pressure = true;
    sa.opt.field_velocity = true;
    sa.opt.field_velocity_from = fp.vf0;
    checked(sa, fp.eval.clone())
}

/// The sub-problem of the pair frequencies oms (theory §12.10): the radiation of every body's
/// rigid and generalized modes with the body pressures (and for a moving body the body velocities,
/// ∇ψ) and the field points; the direct method solves its pairs in it too (Problem::direct).
fn pair_frequency_problem(
    m: &Problem,
    fp: &FieldPoints,
    oms: &[f64],
    moving: bool,
    st: &SharedStatics,
) -> Result<Problem> {
    let mut sb = quiet(m);
    sb.field_statics = Some(st.clone());
    sb.omegas = oms.to_vec();
    rebalance_threads(&mut sb, oms.len());
    sb.headings.clear();
    sb.opt.radiation = true;
    sb.opt.diffraction = false;
    sb.opt.field_pressure = true;
    sb.opt.field_velocity = true;
    sb.opt.field_velocity_from = fp.vf0;
    sb.opt.body_pressure = true;
    sb.opt.body_velocity = moving;
    for b in &mut sb.bodies {
        b.modes = (1..=6).collect();
    }
    checked(sb, fp.eval.clone())
}

/// Whether a body of the model moves (its motions in the forcing and the body condition).
fn moving(m: &Problem) -> bool {
    m.opt.rao && m.bodies.iter().any(|b| !b.modes.is_empty() || !b.generalized.is_empty())
}

/// The sub-problems of the potential QTF of a model as its solve builds them (not solved), for the
/// memory estimate: the first-order one and that of the pair frequencies (none without a pair),
/// with the extra columns of the direct method (its work items; 0 without it), the field points
/// and those with a velocity, whether the indirect method solves the second one first (the
/// radiation) and the arrays of the steps between the solves.
pub(crate) struct SubProblems {
    pub problems: Vec<Problem>,
    pub direct_columns: usize,
    pub field_points: usize,
    pub velocity_points: usize,
    pub moving: bool,
    pub indirect: bool,
    pub footprint: Footprint,
}

/// The arrays of the second order between the sub-problems' solves, for the memory estimate (bytes
/// of the arrays the code allocates, by their element sizes; nb the body points, which the
/// sub-problems' layout gives): the fits, the outer integral, the pairs and what stays held through
/// the direct method's solve. The model's results, the first-order sub-problem's results and the
/// shared field rows are held under all of them and are not counted here.
pub(crate) struct Footprint {
    nf: i64,
    nh: i64,
    n6: i64,
    no: i64,
    p: i64,
    v: i64,
    zone_points: i64, // the points of the quadrature zone's circles and of the outer circle
    zones: i64,       // the coefficients of one fitted zone (Zone)
    work: i64,
    cols: Vec<i64>, // the work items of each pair frequency, the largest first
    acc: i64,
    batch: i64,
    thread: i64, // the temporaries of one work item on its thread
    ind: bool,
    dir: bool,
    moving: bool,
    /// The zones and the field points, held from the start of the second order to its end (bytes):
    /// the panels of the panelled zone (96 each) and the points of the free surface with their
    /// weights (32 each) of Zones; the field points, their evaluation points and wall flags (49 each)
    /// and the near flags and velocity indices of the panelled points (9 each) of FieldPoints; and
    /// the copy of the field points in the sub-problem being built or solved, one at a time (24 each).
    pub(crate) points: i64,
}

/// The working set of the second-order steps after the sub-problems (Footprint::steps), bytes.
pub(crate) struct Steps {
    pub fits: i64,
    pub outer: i64,
    pub pairs: i64,
    pub through_direct: i64,
}

impl Footprint {
    #[allow(non_snake_case)] // the notation of the theory documents
    fn new(s: &Setup) -> Footprint {
        let z = s.z;
        let (nqr, nz, NL, nm, nci) = (z.quad_radii.len(), z.quad_nodes, z.line_points, s.nm, s.ci.rho.len());
        let mut cols: Vec<i64> = s.cols.iter().map(|c| c.len() as i64).collect();
        cols.sort_unstable_by(|a, b| b.cmp(a));
        let quad = (z.pts.len() - z.n_inner) as i64;
        Footprint {
            nf: s.nf as i64,
            nh: s.nh as i64,
            n6: s.n6 as i64,
            no: z.oms.len() as i64,
            p: s.fp.len() as i64,
            v: (s.fp.len() - s.fp.vf0) as i64,
            zone_points: (nqr * nz + NL) as i64,
            // the coefficients on the circles, of the wave and of the far series with its local term
            zones: CD * nm as i64 * (nci as i64 + 1 + 5),
            work: s.work.len() as i64,
            cols,
            acc: outer::accumulated(s) as i64,
            batch: outer::batch(s).1 as i64,
            // the weights of the quadrature zone, the moments, the line weights (items::item)
            thread: CD * (3 * quad + (nci as i64 + 1) * nm as i64 + NL as i64),
            ind: s.ind,
            dir: s.dir,
            moving: moving(s.m),
            points: 96 * z.panels.len() as i64
                + 32 * z.pts.len() as i64
                + (49 + 24) * s.fp.len() as i64
                + 9 * s.fp.ni as i64,
        }
    }

    /// The working set of each step for nb body points, `rb` the bytes of the radiation
    /// sub-problem's results (held during the fits only), the model's frequency_threads and
    /// inner_threads: min(frequency_threads, pair frequencies) workers of the pairs
    /// (FrequencyPools), each a pair frequency at once on its inner threads.
    #[allow(non_snake_case)] // the notation of the theory documents
    pub(crate) fn steps(&self, nb: i64, rb: i64, frequency_threads: i64, inner_threads: i64) -> Steps {
        let (F, H, n6, NO, P, V) = (self.nf, self.nh, self.n6, self.no, self.p, self.v);
        let workers = frequency_threads.max(1).min(NO.max(1));
        let threads = workers * inner_threads.max(1);
        // First: η and its velocity of the disturbances at the field points (complex64), their values
        // on the circles (value and gradient) and the fitted zones, the generalized normals
        let first = CD * 4 * F * P * H + 3 * CD * F * H * self.zone_points + F * H * self.zones + 8 * nb * n6;
        // Radiation (the indirect method): ψ and its velocity at the field points, ψ at the body
        // points, the zones; the motion terms of a moving body hold ψ and ∇ψ on its points
        let mut radiation = 0;
        if self.ind {
            radiation = CD * NO * n6 * (4 * P + nb) + NO * n6 * self.zones;
            if self.moving {
                radiation += 4 * CD * NO * nb * n6;
            }
        }
        let held = first + radiation;
        let outer = CD * self.work * self.acc;
        let coefficients = if self.ind { 2 * CD * F * F * H * H * n6 } else { 0 };
        // the direct method's columns of a work item: the free-surface weights (scalar and gradient),
        // the body condition and its Stokes vectors, the incident wave's force
        let column = if self.dir { CD * (P + 2 * V + nb + if self.moving { 3 * nb } else { 0 } + n6) } else { 0 };
        let columns = self.work * column;
        // a pair frequency at once on each worker: ψ on the circles (the indirect method) and its
        // work items, held until all are done, then copied to the columns
        let at_once: i64 = self.cols.iter().take(workers as usize).sum();
        let per_omega = if self.ind { 3 * CD * n6 * self.zone_points } else { 0 };
        let pairs = held
            + outer
            + coefficients
            + columns
            + workers * per_omega
            + at_once * (column + CD * n6)
            + threads * self.thread;
        // the fits read the velocities from the results (complex32) into complex64: the complex32
        // copy of the largest while it is converted (read_c32)
        let read = 24 * if self.ind { (F * P * H).max(NO * P * n6) } else { F * P * H };
        Steps {
            fits: held + rb + read,
            outer: held + outer + self.batch,
            pairs,
            through_direct: held + outer + coefficients + if self.dir { CD * self.work * n6 } else { 0 },
        }
    }
}

/// The sub-problems of the potential QTF of the validated model m (SubProblems).
pub(crate) fn sub_problems(m: &Problem) -> Result<SubProblems> {
    let z = zones(m)?;
    let s = Setup::new(m, &z);
    let st = SharedStatics::default();
    let mv = moving(m);
    let mut problems = vec![first_order_problem(m, &s.fp, &s.omegas, mv, &st)?];
    if !z.oms.is_empty() {
        problems.push(pair_frequency_problem(m, &s.fp, &z.oms, mv, &st)?);
    }
    Ok(SubProblems {
        problems,
        direct_columns: if s.dir { s.work.len() } else { 0 },
        field_points: s.fp.len(),
        velocity_points: s.fp.len() - s.fp.vf0,
        moving: mv,
        indirect: s.ind,
        footprint: Footprint::new(&s),
    })
}

// ---------------------------------------------------------------- the field points

/// The field points of the sub-problems (theory §12.8-§12.10): the panelled zone's points (the
/// first ni), the circles of the fit (nr points each), four copies of the panelled points moved by
/// ±step in x and y (the horizontal gradient by central differences of the potential: the field
/// potential of Green's identity converges with the square of the panel size, the source
/// formulation's field velocity only linearly, theory §4.2), for the direct method the waterline
/// sides' midpoints (moved off the body by 1e-6 R), and last the points next to a waterline once
/// more (within three steps of a waterline side: the field velocity there, where the stencil may
/// reach a body and the field potential is inaccurate so close to the panels; the sub-problems
/// give the velocities at these last points only). With a wall the points beyond it are evaluated
/// at their mirror images (the fields are even about the wall: the potentials equal, the
/// velocities mirrored, flip).
struct FieldPoints {
    field: Vec<Vec3>,
    eval: Vec<Vec3>,
    flip: Vec<bool>,
    near: Vec<bool>, // a panelled point next to a waterline
    vat: Vec<usize>, // the field index of its velocity
    step: f64,
    ni: usize,  // the panelled points
    nr: usize,  // the points of a circle of the fit
    st0: usize, // the first stencil copy
    wl0: usize, // the first waterline midpoint (the direct method)
    vf0: usize, // the first point with a velocity
    wall: Option<usize>,
}

impl FieldPoints {
    #[allow(non_snake_case)] // the notation of the theory documents
    fn new(z: &Zones, rings: &[f64], direct: bool) -> FieldPoints {
        let (c0, pts, Ni, NR) = (z.c0, &z.pts[..], z.n_inner, z.ring_points);
        let Rin = z.inner_radius;
        // the stencil step of the central differences (theory §12.8, gradients): differences
        // of the field potential, which converges with the square of the panel size, rather than
        // the field velocity, which converges only linearly
        let step = (crate::env_hooks::validation_constant("STENCIL_WAVENUMBER") / z.ks_max)
            .min(crate::env_hooks::validation_constant("STENCIL_RADIUS") * Rin);
        let near: Vec<bool> = pts[..Ni]
            .iter()
            .map(|(x, _)| {
                z.wsegs_all.iter().any(|&(a, b)| {
                    let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
                    (x[0] - mid[0]).hypot(x[1] - mid[1]) < 3.0 * step + 0.5 * (b[0] - a[0]).hypot(b[1] - a[1])
                })
            })
            .collect();
        // at their length (the memory estimate, Footprint::points): the panelled points and their
        // four stencil copies, the rings, the waterline midpoints, the velocity points
        let nv = near.iter().filter(|&&b| b).count();
        let mut field: Vec<Vec3> =
            Vec::with_capacity(5 * Ni + rings.len() * NR + if direct { z.wsegs.len() } else { 0 } + nv);
        for (x, _) in &pts[..Ni] {
            field.push(*x);
        }
        for &rr in rings {
            for t in 0..NR {
                let th = 2.0 * PI * t as f64 / NR as f64;
                field.push(v3(c0[0] + rr * th.cos(), c0[1] + rr * th.sin(), 0.0));
            }
        }
        for d in [[step, 0.0], [-step, 0.0], [0.0, step], [0.0, -step]] {
            for (x, _) in &pts[..Ni] {
                field.push(v3(x[0] + d[0], x[1] + d[1], 0.0));
            }
        }
        let wl0 = field.len();
        if direct {
            let off = crate::env_hooks::validation_constant("WATERLINE_OFFSET") * Rin;
            for s in &z.wsegs {
                let (a, b) = (s.p[0], s.p[1]);
                let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
                let (dx, dy) = (mid[0] - c0[0], mid[1] - c0[1]);
                let d = dx.hypot(dy).max(1e-300);
                field.push(v3(mid[0] + off * dx / d, mid[1] + off * dy / d, 0.0));
            }
        }
        let vf0 = field.len();
        let mut vat = vec![usize::MAX; Ni];
        for (q, (x, _)) in pts[..Ni].iter().enumerate() {
            if near[q] {
                vat[q] = field.len();
                field.push(*x);
            }
        }
        let wall = z.wall;
        let flip: Vec<bool> = field.iter().map(|x| wall.is_some_and(|ax| x[ax] < 0.0)).collect();
        let eval = field
            .iter()
            .zip(&flip)
            .map(|(x, &f)| {
                let mut y = *x;
                if let (Some(ax), true) = (wall, f) {
                    y[ax] = -y[ax];
                }
                y
            })
            .collect();
        FieldPoints { field, eval, flip, near, vat, step, ni: Ni, nr: NR, st0: Ni + rings.len() * NR, wl0, vf0, wall }
    }
    /// The field points.
    fn len(&self) -> usize {
        self.field.len()
    }
    /// The velocities of field point q ≥ vf0 at its mirror image, mirrored back: the component
    /// normal to the wall changes sign.
    fn mirror_v(&self, v: &mut [Complex64], q: usize) {
        if let (Some(ax), true) = (self.wall, self.flip[q]) {
            for c in v.chunks_mut(3) {
                c[ax] = -c[ax];
            }
        }
    }
}

// ---------------------------------------------------------------- the set-up of a solve

/// A work item: the pair (a, ha; b, hb) and whether it is the sum (else the difference, Ω > 0).
type Pair = (usize, usize, usize, usize, bool);

/// What every phase of the potential QTF of a model shares: the model and its zones, the waves,
/// the circles of the fit and the field points, the work items (the pairs with Ω ≥ 0: the
/// difference of the larger frequency first) by pair frequency, the tables of e^{imθ} of the
/// quadrature zone's angles and the outer circle's points, the methods asked for.
#[allow(non_snake_case)] // the notation of the theory documents
struct Setup<'a> {
    m: &'a Problem,
    z: &'a Zones,
    g: f64,
    h: f64,
    omegas: Vec<f64>,
    heads: Vec<f64>,
    nf: usize,
    nh: usize,
    n6: usize,
    hf: f64, // with a wall the whole plane is integrated: its area and outer-circle terms halved
    ci: Circles,
    rings: Vec<f64>,
    fp: FieldPoints,
    work: Vec<Pair>,
    cols: Vec<Vec<usize>>, // the work items of each pair frequency, in the order of the work
    used: Vec<bool>,       // the pair frequencies the work needs
    nm: usize,             // the orders of the fitted series, 2 Mf + 1
    qe: Vec<Complex64>,
    qcs: Vec<(f64, f64)>,
    le: Vec<Complex64>,
    lcs: Vec<(f64, f64)>,
    ind: bool,
    dir: bool,
}

impl<'a> Setup<'a> {
    #[allow(non_snake_case)] // the notation of the theory documents
    fn new(m: &'a Problem, z: &'a Zones) -> Setup<'a> {
        let (ind, dir) = if m.opt.potential_qtf_both {
            (true, true)
        } else {
            (!m.opt.potential_qtf_direct, m.opt.potential_qtf_direct)
        };
        let omegas: Vec<f64> = m.omegas.iter().copied().filter(|w| *w > 0.0 && w.is_finite()).collect();
        let heads: Vec<f64> = m.headings.iter().map(|d| d * PI / 180.0).collect();
        let (F, H) = (omegas.len(), heads.len());
        let n6: usize = m.bodies.iter().map(|b| 6 + b.generalized.len()).sum();
        let ci = Circles::new(z.inner_radius, z.outer_radius, z.circles);
        let rings: Vec<f64> = ci.rho.iter().copied().chain([far_field::fit_ratio() * z.outer_radius]).collect();
        let fp = FieldPoints::new(z, &rings, dir);
        let pairs_h = m.opt.potential_qtf == QtfHeadings::Pairs;
        let mut work: Vec<Pair> = Vec::new();
        for a in 0..F {
            for bb in 0..F {
                for ha in 0..H {
                    for hb in 0..H {
                        if !pairs_h && ha != hb {
                            continue;
                        }
                        if z.want_sum && a <= bb && (a < bb || ha <= hb) {
                            work.push((a, bb, ha, hb, true)); // sum
                        }
                        if z.want_dif && omegas[a] > omegas[bb] {
                            work.push((a, bb, ha, hb, false)); // difference
                        }
                    }
                }
            }
        }
        let NO = z.oms.len();
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); NO];
        let mut used = vec![false; NO];
        let oidx = |o: f64| z.oms.iter().position(|x| (x - o).abs() <= 1e-9 * o.max(1e-300));
        for (wi, &(a, bb, _, _, sum)) in work.iter().enumerate() {
            let om = if sum { omegas[a] + omegas[bb] } else { omegas[a] - omegas[bb] };
            let o = oidx(om).expect("a pair frequency");
            cols[o].push(wi);
            used[o] = true;
        }
        let Mf = z.fit_order;
        let nm = 2 * Mf + 1;
        // e^{imθ} (|m| <= Mf) of the quadrature zone's angles (the same on each of its circles) and
        // of the outer circle's points
        let table = |th: &[f64]| -> Vec<Complex64> {
            th.iter()
                .flat_map(|&t| (0..nm).map(move |i| Complex64::from_polar(1.0, (i as f64 - Mf as f64) * t)))
                .collect()
        };
        let cs_of = |th: &[f64]| -> Vec<(f64, f64)> { th.iter().map(|t| (t.cos(), t.sin())).collect() };
        let nz = z.quad_nodes;
        let qth: Vec<f64> = (0..nz).map(|t| 2.0 * PI * (t as f64 + 0.5) / nz as f64).collect();
        let lth: Vec<f64> = (0..z.line_points).map(|t| z.line_angle(t)).collect();
        Setup {
            m,
            z,
            g: m.env.gravity,
            h: m.env.depth,
            hf: if z.wall.is_some() { 0.5 } else { 1.0 },
            omegas,
            heads,
            nf: F,
            nh: H,
            n6,
            ci,
            rings,
            fp,
            work,
            cols,
            used,
            nm,
            qe: table(&qth),
            qcs: cs_of(&qth),
            le: table(&lth),
            lcs: cs_of(&lth),
            ind,
            dir,
        }
    }

    /// The first-order wave of frequency f and heading hh.
    fn wave(&self, f: usize, hh: usize) -> Wave {
        let w = self.omegas[f];
        Wave {
            w,
            k: self.z.kf[f],
            kk: w * w / self.g,
            beta: self.heads[hh],
            image: self.z.wall.map(|ax| mirror_heading(ax, self.heads[hh])),
        }
    }

    /// The pair frequency of work item wi.
    fn omega(&self, wi: usize) -> f64 {
        let (a, bb, _, _, sum) = self.work[wi];
        if sum { self.omegas[a] + self.omegas[bb] } else { self.omegas[a] - self.omegas[bb] }
    }

    /// The index of the pair frequency o among the distinct ones.
    fn oidx(&self, o: f64) -> usize {
        self.z.oms.iter().position(|x| (x - o).abs() <= 1e-9 * o.max(1e-300)).expect("a pair frequency")
    }

    /// The far field of a field given at the field points (v at a field index) beyond the panelled
    /// zone, from the discrete Fourier coefficients (to the order Mf) of its values on the circles
    /// of the fit: k its wavenumber, w its frequency, den the wave denominators of k.
    fn fit(&self, v: &dyn Fn(usize) -> Complex64, den: &[Complex64], k: f64, w: f64) -> Zone {
        let (ni, nr, mf) = (self.fp.ni, self.fp.nr, self.z.fit_order);
        let c = (0..self.rings.len()).map(|i| ring_dft(&(0..nr).map(|t| v(ni + i * nr + t)).collect::<Vec<_>>(), mf));
        Zone::fit(c.collect(), &self.ci, den, mf, k, w * w / self.g, self.h)
    }

    /// A field on the points of the quadrature zone's circle c (rad its radial functions there).
    fn in_zone(&self, zf: &Zone, rad: &ZoneRadial, c: usize) -> Vec<(Complex64, [Complex64; 2])> {
        on_circle(&zf.modes(rad), self.z.quad_radii[c], &self.qe, &self.qcs)
    }

    /// A field on the points of the outer circle (its series).
    fn on_outer(&self, zf: &Zone) -> Vec<(Complex64, [Complex64; 2])> {
        let (r, mf) = (self.z.outer_radius, self.z.fit_order);
        on_circle(&zf.far.modes(&zf.far.radial(r, r, mf), self.g), r, &self.le, &self.lcs)
    }

    /// The radial functions of the quadrature zone at the wavenumber k on each of its circles.
    fn zone_radial(&self, k: f64) -> Vec<ZoneRadial> {
        let mf = self.z.fit_order;
        self.z.quad_radii.par_iter().map(|&r| Zone::radial(&self.ci, k, r, mf)).collect()
    }
}

// ---------------------------------------------------------------- the first-order fields

/// The first-order fields of the model's waves at the free surface and on the bodies: the
/// sub-problem's results, the disturbance (the scattering and the radiation of the motions of every
/// body) at the field points as η [F][P][H] (per unit amplitude) and its velocity [F][P][H][3]
/// (from vf0 on), the motions ξ [F][H][n6] (zero for the modes left out), the body layout, the
/// waterline values, the far fields of the disturbances (their zones, their values on the
/// quadrature zone's circles [(f H + h) nqr + circle][t] and on the outer circle [f H + h][t], the
/// series truncated at M) and the generalized normals of the body points.
struct First {
    ra: Box<Results>,
    eta: Vec<Complex64>,
    vel: Vec<Complex64>,
    xi: Vec<Complex64>,
    lay: Layout,
    wlf: Vec<WlFs>,
    dq: Vec<Vec<(Complex64, [Complex64; 2])>>,
    dl: Vec<Vec<(Complex64, [Complex64; 2])>>,
    far: Vec<Far>,
    njs: Vec<Vec<f64>>,
}

impl First {
    #[allow(non_snake_case)] // the notation of the theory documents
    fn new(s: &Setup, ra: Box<Results>, moving: bool) -> Result<First> {
        let (m, F, H, n6, P) = (s.m, s.nf, s.nh, s.n6, s.fp.len());
        let vf0 = s.fp.vf0;
        let lay = Layout::new(m, &ra, moving)?;
        let mut eta = read_c32(&ra, QuantityId::FieldPointsPressureScattering); // [F][P][H]
        let mut vel = read_c32(&ra, QuantityId::FieldPointsVelocityScattering); // [F][P][H][3], from vf0 on
        if eta.len() != F * P * H || vel.len() != 3 * F * P * H {
            return Err(Error::internal("potential QTF: unexpected sizes of the first-order results"));
        }
        let mut xi = vec![CZ; F * H * n6];
        if moving {
            let (nc, free) = (lay.nc, &lay.free);
            let mot = read_c32(&ra, QuantityId::Motion); // [F][H][n6]
            let fpr = read_c32(&ra, QuantityId::FieldPointsPressureRadiation); // [F][P][nc]
            let fvr = read_c32(&ra, QuantityId::FieldPointsVelocityRadiation); // [F][P][nc][3]
            if mot.len() != F * H * n6 || fpr.len() != F * P * nc || fvr.len() != 3 * F * P * nc {
                return Err(Error::internal("potential QTF: unexpected sizes of the first-order results"));
            }
            for f in 0..F {
                for hh in 0..H {
                    for fr in free {
                        for &(j, _) in fr {
                            xi[(f * H + hh) * n6 + j] = mot[(f * H + hh) * n6 + j];
                        }
                    }
                }
            }
            for f in 0..F {
                for q in 0..P {
                    for hh in 0..H {
                        for fr in free {
                            for &(j, cc) in fr {
                                let x = xi[(f * H + hh) * n6 + j];
                                eta[(f * P + q) * H + hh] += x * fpr[(f * P + q) * nc + cc];
                                if q >= vf0 {
                                    for d in 0..3 {
                                        vel[((f * P + q) * H + hh) * 3 + d] += x * fvr[((f * P + q) * nc + cc) * 3 + d];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for f in 0..F {
            for q in vf0..P {
                let o = (f * P + q) * H * 3;
                s.fp.mirror_v(&mut vel[o..o + 3 * H], q);
            }
        }
        let wlf = free_surface_waterline(m, &ra, &lay, &s.z.wsegs, &s.omegas, &xi, H)?;
        let njs = lay.generalized_normals(m);
        let mut first = First { ra, eta, vel, xi, lay, wlf, dq: Vec::new(), dl: Vec::new(), far: Vec::new(), njs };
        // the far fields of the disturbances
        let kf = &s.z.kf;
        let mf = s.z.fit_order;
        let den: Vec<Vec<Complex64>> = kf.iter().map(|&k| s.ci.wave_denominators(k, mf)).collect();
        let zones: Vec<Zone> = (0..F * H)
            .into_par_iter()
            .map(|fh| {
                let (f, hh) = (fh / H, fh % H);
                s.fit(&|q| first.pot(s, f, q, hh), &den[f], kf[f], s.omegas[f])
            })
            .collect();
        let nqr = s.z.quad_radii.len();
        let radf: Vec<Vec<ZoneRadial>> = kf.iter().map(|&k| s.zone_radial(k)).collect();
        first.dq = (0..F * H * nqr)
            .into_par_iter()
            .map(|i| {
                let (fh, c) = (i / nqr, i % nqr);
                s.in_zone(&zones[fh], &radf[fh / H][c], c)
            })
            .collect();
        first.dl = zones.par_iter().map(|zf| s.on_outer(zf)).collect();
        // beyond the outer circle the series truncated at M (the asymptotic zone)
        first.far = zones.iter().map(|zf| zf.far.truncated(s.z.order)).collect();
        Ok(first)
    }

    /// The disturbance's potential at field point q of (frequency, heading).
    fn pot(&self, s: &Setup, f: usize, q: usize, hh: usize) -> Complex64 {
        self.eta[(f * s.fp.len() + q) * s.nh + hh] * (I * s.g / s.omegas[f])
    }

    /// The disturbance's potential and gradient at a point of the panelled zone (q < ni): ∇_h by
    /// central differences of the potential (the field velocity next to a waterline), ∂_z = Kφ (the
    /// free-surface condition).
    fn sc(&self, s: &Setup, f: usize, q: usize, hh: usize) -> (Complex64, [Complex64; 3]) {
        let fp = &s.fp;
        let phi = self.pot(s, f, q, hh);
        if fp.near[q] {
            let o = ((f * fp.len() + fp.vat[q]) * s.nh + hh) * 3;
            return (phi, [self.vel[o], self.vel[o + 1], self.vel[o + 2]]);
        }
        let d = |k: usize| self.pot(s, f, fp.st0 + k * fp.ni + q, hh);
        let kk = s.omegas[f] * s.omegas[f] / s.g;
        (phi, [(d(0) - d(1)) / (2.0 * fp.step), (d(2) - d(3)) / (2.0 * fp.step), kk * phi])
    }

    /// The motion terms of every moving body, and of a body with a moving plate over its waterline
    /// (rb: the radiation at Ω, the indirect method's ψ).
    fn motions(&self, s: &Setup, rb: Option<&Results>, moving: bool) -> Result<Vec<Motion>> {
        let mut out = Vec::new();
        if !moving {
            return Ok(out);
        }
        for bi in 0..s.m.bodies.len() {
            let sides = &s.z.sides[bi];
            if body::has_motion_terms(s.m, bi, sides) {
                out.push(Motion::new(s.m, bi, sides, &self.lay, &self.ra, rb, &self.xi, &s.omegas, &s.z.oms, s.nh)?);
            }
        }
        Ok(out)
    }
}

/// The radiation at the pair frequencies (the indirect method): ψ per unit velocity at the field
/// points [O][P][n6] (η per unit displacement = (Ω²/g) ψ), its velocity [O][P][n6][3] (from vf0 on,
/// mirrored beyond a wall), ψ at the body points [O][Nb][n6], and its far fields (the zones and the
/// series truncated at M) [O][n6].
struct Radiation {
    fpr: Vec<Complex64>,
    fvb: Vec<Complex64>,
    bpr: Vec<Complex64>,
    zones: Vec<Zone>,
    far: Vec<Far>,
}

impl Radiation {
    #[allow(non_snake_case)] // the notation of the theory documents
    fn new(s: &Setup, rb: &Results, nb: usize) -> Result<Radiation> {
        let (n6, P, NO) = (s.n6, s.fp.len(), s.z.oms.len());
        let fpr = read_c32(rb, QuantityId::FieldPointsPressureRadiation); // [O][P][n6]
        let mut fvb = read_c32(rb, QuantityId::FieldPointsVelocityRadiation); // [O][P][n6][3], from vf0 on
        let bpr = read_c32(rb, QuantityId::BodyPointsPressureRadiation); // [O][Nb][n6]
        if fpr.len() != NO * P * n6 || fvb.len() != 3 * NO * P * n6 || bpr.len() != NO * nb * n6 {
            return Err(Error::internal("potential QTF: unexpected sizes of the sub-problems' results"));
        }
        for o in 0..NO {
            for q in s.fp.vf0..P {
                let i = (o * P + q) * n6 * 3;
                s.fp.mirror_v(&mut fvb[i..i + 3 * n6], q);
            }
        }
        let mut r = Radiation { fpr, fvb, bpr, zones: Vec::new(), far: Vec::new() };
        let mf = s.z.fit_order;
        let den: Vec<Vec<Complex64>> = s.z.ko.iter().map(|&k| s.ci.wave_denominators(k, mf)).collect();
        r.zones = (0..NO * n6)
            .into_par_iter()
            .map(|oj| {
                let (o, j) = (oj / n6, oj % n6);
                s.fit(&|q| r.psi_f(s, o, q, j), &den[o], s.z.ko[o], s.z.oms[o])
            })
            .collect();
        r.far = r.zones.iter().map(|zf| zf.far.truncated(s.z.order)).collect();
        Ok(r)
    }

    /// ψ_j at field point q of pair frequency o.
    fn psi_f(&self, s: &Setup, o: usize, q: usize, j: usize) -> Complex64 {
        let w = s.z.oms[o];
        self.fpr[(o * s.fp.len() + q) * s.n6 + j] * (s.g / (w * w))
    }

    /// ψ_j at body point q of pair frequency o.
    fn psi_b(&self, s: &Setup, nb: usize, o: usize, q: usize, j: usize) -> Complex64 {
        let w = s.z.oms[o];
        self.bpr[(o * nb + q) * s.n6 + j] * (s.g / (w * w))
    }

    /// ∇_h ψ_j at a point of the panelled zone: central differences of ψ (the field velocity per
    /// unit displacement / (iΩ) next to the waterlines).
    fn gpsi_f(&self, s: &Setup, o: usize, q: usize, j: usize) -> [Complex64; 2] {
        let fp = &s.fp;
        if fp.near[q] {
            let w = s.z.oms[o];
            let i = ((o * fp.len() + fp.vat[q]) * s.n6 + j) * 3;
            return [self.fvb[i] / (I * w), self.fvb[i + 1] / (I * w)];
        }
        let d = |k: usize| self.psi_f(s, o, fp.st0 + k * fp.ni + q, j);
        [(d(0) - d(1)) / (2.0 * fp.step), (d(2) - d(3)) / (2.0 * fp.step)]
    }
}

// ---------------------------------------------------------------- the solve

/// Adds the potential QTFs (theory §12.1; theory §12.4-§12.15) of the methods the model asks for to
/// its results r: the two sub-problems (solved with the cancel request of the model's solve, their
/// progress their own), the fits, the outer integral, the inner free surface and the bodies of
/// every pair and for the direct method its solve, on frequency_threads × inner_threads threads.
pub fn add_potential_qtfs(m: &Problem, ctx: &SolveContext, r: &mut Results) -> Result<()> {
    let pool = crate::parallel::pool("qtf", m.opt.total_threads())?;
    pool.install(|| run(m, ctx, r))
}

#[allow(non_snake_case)] // the notation of the theory documents
fn run(m: &Problem, ctx: &SolveContext, r: &mut Results) -> Result<()> {
    let mut prof = Profile::new(); // phase timing on stderr with HB_PROFILE
    let z = zones(m)?;
    let s = Setup::new(m, &z);
    let mv = moving(m);
    if prof.on {
        eprintln!(
            "hb profile: potential QTF: {} free-surface points ({} panelled), {} field points, {} pair frequencies, M = {}, fitted to {}",
            z.pts.len(),
            z.n_inner,
            s.fp.len(),
            z.oms.len(),
            z.order,
            z.fit_order
        );
    }
    // the sub-problems: the cancel request of the model's solve, their progress as steps of its
    // second order (their frequencies the items); one set of field points, their static rows shared
    let st = SharedStatics::default();
    let mut sa = first_order_problem(m, &s.fp, &s.omegas, mv, &st)?;
    prof.mark("QTF zones and field points");
    let ra = sub_solve(ctx, SecondOrderStep::PotentialFirstOrder, &mut sa)?;
    drop(sa);
    prof.mark("QTF first order");
    // the pair frequencies: the indirect method needs ψ at Ω first; the direct method solves this
    // sub-problem after the pairs, with its extra columns (its weights need no ψ)
    let mut sb = if z.oms.is_empty() { None } else { Some(pair_frequency_problem(m, &s.fp, &z.oms, mv, &st)?) };
    let rb = match (&mut sb, s.ind) {
        (Some(sb), true) => Some(sub_solve(ctx, SecondOrderStep::PotentialRadiation, sb)?),
        _ => None,
    };
    prof.mark("QTF radiation at Omega");
    // the fits of the fields on the circles (on all threads, one item)
    let fits = ctx.step(SecondOrderStep::PotentialFits, 1);
    let it = fits.item(0, None);
    let fitted = (|| -> Result<_> {
        it.check_cancel()?;
        let first = First::new(&s, ra, mv)?;
        let nb = first.lay.len();
        let motions = first.motions(&s, rb.as_deref(), mv)?;
        let rad = rb.as_deref().map(|rb| Radiation::new(&s, rb, nb)).transpose()?;
        Ok((first, motions, rad))
    })();
    it.end(&fitted);
    fits.end(fitted.is_ok());
    let (first, motions, rad) = fitted?;
    drop(rb);
    prof.mark("QTF far-field fits");
    ctx.check_cancel()?;
    let outer = outer::integral(&s, &first, rad.as_ref(), ctx)?;
    prof.mark("QTF outer integral");
    let done = items::pairs(&s, &first, &motions, rad.as_ref(), &outer, ctx)?;
    prof.mark("QTF inner and body");
    let direct = match (done.direct, &mut sb) {
        (Some(d), Some(sb)) => Some(items::direct_solve(&s, &first, d, sb, ctx)?),
        _ => None,
    };
    prof.mark("QTF direct solve");
    if s.ind {
        put_method(&s, r, false, done.indirect.as_ref().map(|(d, x)| (&d[..], &x[..])));
    }
    if s.dir {
        put_method(&s, r, true, direct.as_ref().map(|(d, x)| (&d[..], &x[..])));
    }
    crate::heading_pairs::put_pairs(
        r,
        QuantityId::QtfHeadingPairs,
        &m.headings,
        &crate::heading_pairs::heading_pairs(s.nh, m.opt.qtf_pairs),
    );
    prof.mark("QTF output");
    Ok(())
}

/// A sub-problem solved as the second-order step `step` of the model's solve (ctx): its limits
/// and frequencies the items of the step, on the model's frequency workers.
fn sub_solve(ctx: &SolveContext, step: SecondOrderStep, p: &mut Problem) -> Result<Box<Results>> {
    let (limits, n) = crate::model_solve::board_frequencies(p);
    let sp = ctx.sub_problem(step, limits, n);
    let r = solve(p, &sp.ctx);
    sp.end(r.is_ok());
    r
}

/// The coefficients [F][F][H][H][n6] of a method (difference, sum) under its quantities: the
/// heading pairs of the QTFs on the coordinate axis, the rigid rows (6 per body) and the generalized
/// forces of the generalized modes at their coordinates. None: no pair (NaN but zero at equal
/// frequencies).
#[allow(non_snake_case)] // the notation of the theory documents
fn put_method(s: &Setup, r: &mut Results, direct: bool, v: Option<(&[Complex64], &[Complex64])>) {
    let (m, F, H, n6) = (s.m, s.nf, s.nh, s.n6);
    let (dif, sm) = match v {
        Some((d, x)) => (d.to_vec(), x.to_vec()),
        None => items::empty(s),
    };
    // the first row of each body among the n6 rows (6 + its generalized modes a body): its first
    // coordinate, as the potential QTF takes no pressure modes (refused by the problem rules)
    debug_assert!(m.bodies.iter().all(|b| b.pressure.is_empty()));
    let off = m.first_coordinates();
    let hp = crate::heading_pairs::heading_pairs(H, m.opt.qtf_pairs);
    let n6r = 6 * m.bodies.len();
    let rows = crate::heading_pairs::CoordinateRows::of(m);
    let dims = vec![F as i64, F as i64, hp.len() as i64, rows.d as i64];
    let (qd, qs) = if direct {
        (QuantityId::QtfPotentialDirectDifference, QuantityId::QtfPotentialDirectSum)
    } else {
        (QuantityId::QtfPotentialIndirectDifference, QuantityId::QtfPotentialIndirectSum)
    };
    // the rows of the loads apart: the six rigid ones of each body and its generalized ones
    let gt = rows.generalized.len();
    let split = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
        if gt == 0 {
            return (v.to_vec(), Vec::new());
        }
        let (mut r6, mut rg) = (Vec::new(), Vec::new());
        for row in v.chunks(n6) {
            for (bi, &k) in off.iter().enumerate() {
                r6.extend_from_slice(&row[k..k + 6]);
                rg.extend_from_slice(&row[k + 6..k + 6 + m.bodies[bi].generalized.len()]);
            }
        }
        (r6, rg)
    };
    let ((d6, dg), (s6, sg)) = (split(&dif), split(&sm));
    let pick = |v: &[Complex64], n: usize| {
        if n == 0 { Vec::new() } else { crate::heading_pairs::select_pairs(v, F * F, H, n, &hp) }
    };
    put_vec(r, qd, dims.clone(), rows.place(pick(&d6, n6r), &pick(&dg, gt)));
    put_vec(r, qs, dims, rows.place(pick(&s6, n6r), &pick(&sg, gt)));
}
