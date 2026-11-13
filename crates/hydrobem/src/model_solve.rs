//! The solve of a model (model.rs): `Model::solve`, `Model::begin_solve` and `Model::finish_solve`,
//! and the `SolveJob` that owns the validated problem while it runs (maybe on another thread), with
//! the quantities a solve adds to its results (the hydrostatics of every body, the coordinates of
//! the D axis, the field outputs split into those of the sea and those of the tanks).

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::codes::{QuantityId, Warning};
use crate::document::Document;
use crate::document_build::{sea_points, tank_points};
use crate::document_enabled::Numbering;
use crate::env_hooks::{ActiveHook, environment_hooks};
use crate::error::{Diagnostic, Error, ErrorKind, Location, Result};
use crate::memory::check_memory;
use crate::model::Model;
use crate::model_codes::NewmanDrifts;
use crate::model_types::{Progress, ProgressFn};
use crate::problem::{CancelFlag, Problem};
use crate::progress::{Board, CancelToken, Limit, Phase, SolveContext};
use crate::response::add_momentum_drift;
use crate::results::{Quantity, QuantityData, Results, put};
use crate::results_file::results_meta;
use crate::solver::solve;

impl Model {
    /// Validates and solves the model; its results are kept in it. progress is called for
    /// the stages and after each frequency, from one thread at a time. ErrorKind::Cancelled after a cancel
    /// of the token, ErrorKind::Numerical for a singular system, ErrorKind::Value before the start when
    /// the memory estimate exceeds the available memory (options.memory_check).
    pub fn solve<F: FnMut(&Progress) + Send>(&mut self, mut progress: F) -> Result<&Results> {
        let mut job = self.begin_solve()?;
        let outcome = job.run(Some(&mut progress));
        self.finish_solve(job, outcome)
    }
    /// The checks before a solve, the same for `Model::check` and `begin_solve`, in this order: the
    /// problem built from the enabled objects (the rules between the objects; its errors located in
    /// the model), the model rules (`Problem::check_rules`), the route's refusals
    /// (ErrorKind::Unsupported) and, with the potential QTF, the zones of its free surface as the
    /// solve makes them (`potential_qtf::zones`: their refusals, reference/inputs-orders §3). The
    /// errors after the build are those of the problem: `renumbered` locates them in the model.
    pub(crate) fn pre_solve(&mut self) -> Result<()> {
        self.build()?;
        self.problem.validate()?;
        if self.problem.opt.potential_qtf.any() {
            crate::potential_qtf::zones(&self.problem)?;
        }
        Ok(())
    }
    /// The start of a solve: the model is calculating from now on, its results are gone, and its
    /// problem is built and passes the checks before a solve (`pre_solve`, as `Model::check`: a
    /// free surface of the potential QTF that cannot be built is refused here, before the solve)
    /// and moves into the job (ErrorKind::State when it is calculating already). The model can be
    /// read while the job runs; finish_solve takes its outcome. The hooks of the environment are
    /// read here (ErrorKind::Value for one that is refused), and with options.memory_check the
    /// memory estimate is checked here.
    pub fn begin_solve(&mut self) -> Result<SolveJob> {
        if self.calculating.swap(true, Ordering::SeqCst) {
            return Err(Error::state("the model is calculating already"));
        }
        // the results of earlier inputs, or of a failed or cancelled solve, are gone
        self.results = None;
        self.results_revision = u64::MAX;
        let started = (|| -> Result<Vec<ActiveHook>> {
            // a hook of the environment that is set is recorded in the results; one that is refused
            // refuses the solve
            let hooks = environment_hooks()?;
            self.pre_solve()?;
            // a solve that needs more memory than is available is refused here, before its first
            // allocation could fail and abort the process
            if self.problem.opt.memory_check {
                check_memory(&mut self.problem)?;
            }
            Ok(hooks)
        })();
        let hooks = match started {
            Ok(h) => h,
            Err(e) => {
                // the end of the solve: the state and the cancel request cleared
                self.calculating.store(false, Ordering::SeqCst);
                self.cancel.clear();
                return Err(self.renumbered(e));
            }
        };
        let problem = std::mem::take(&mut self.problem);
        self.problem.cancel = CancelFlag::shared(self.cancel.clone());
        // the board of the progress: the limits (radiation only), then the ordinary frequencies,
        // and the frequency workers
        let (limits, ordinary) = board_frequencies(&problem);
        let board = self.monitor.begin(limits, ordinary, problem.opt.frequency_threads.max(1) as usize);
        let built = std::mem::replace(&mut self.built_revision, u64::MAX);
        Ok(SolveJob {
            problem,
            built_revision: built,
            numbering: self.numbering(),
            doc: Arc::new(self.doc.clone()),
            calculating: self.calculating.clone(),
            cancel: self.cancel.clone(),
            board,
            hooks,
        })
    }
    /// The end of a solve: the problem back in the model and, for a solve that succeeded, its
    /// results kept; the state and the cancel request are cleared (also for a failed or
    /// cancelled solve).
    pub fn finish_solve(&mut self, mut job: SolveJob, outcome: Result<Results>) -> Result<&Results> {
        self.problem = std::mem::take(&mut job.problem);
        self.built_revision = job.built_revision;
        let numbering = std::mem::take(&mut job.numbering);
        job.board.end(end_phase(&outcome));
        drop(job); // the state and the cancel request cleared
        let r = outcome.map_err(|e| numbering.error(e))?;
        self.results = Some(Arc::new(r));
        self.results_revision = self.doc.revision();
        Ok(self.results.as_deref().expect("results"))
    }
}

/// The limits (radiation only) and the number of ordinary frequencies of a solve of the problem, as
/// its frequencies run (solve_driver::run_frequencies): the slots of its progress board (none for
/// the hydrostatics alone, which have no frequencies).
pub(crate) fn board_frequencies(problem: &Problem) -> (Vec<Limit>, usize) {
    if problem.hydrostatics_only() {
        return (Vec::new(), 0);
    }
    let mut limits = Vec::new();
    if problem.opt.radiation && problem.has_zero() {
        limits.push(Limit::Zero);
    }
    if problem.opt.radiation && problem.has_infinite() {
        limits.push(Limit::Infinite);
    }
    (limits, problem.omegas.iter().filter(|w| **w > 0.0 && w.is_finite()).count())
}

/// The end phase of a solve of this outcome: done, cancelled or failed.
fn end_phase<T>(r: &Result<T>) -> Phase {
    match r {
        Ok(_) => Phase::Done,
        Err(e) if e.kind() == ErrorKind::Cancelled => Phase::Cancelled,
        Err(_) => Phase::Failed,
    }
}

/// The coordinates of the D axis: body, kind (0 rigid, 1 generalized, 2 pressure), the
/// one-based index within the body, 1 for a rotation.
pub(crate) fn add_coordinates(m: &Problem, r: &mut Results) {
    let mut c: Vec<i32> = Vec::new();
    for (b, body) in m.bodies.iter().enumerate() {
        let bi = b as i32;
        for j in 1..=6 {
            c.extend_from_slice(&[bi, 0, j, if j >= 4 { 1 } else { 0 }]);
        }
        for g in 0..body.generalized.len() {
            c.extend_from_slice(&[bi, 1, g as i32 + 1, 0]);
        }
        let mut k = 0;
        for ps in &body.pressure {
            for _ in 0..ps.modes.len() {
                k += 1;
                c.extend_from_slice(&[bi, 2, k, 0]);
            }
        }
    }
    put(r, QuantityId::Coordinates, vec![(c.len() / 4) as i64, 4], &c);
}

// the hydrostatics of every body as quantities of the results (the names of the query)
fn add_hydrostatics(m: &Problem, r: &mut Results) {
    let nb = m.bodies.len() as i64;
    let (mut vols, mut vol, mut dm, mut cb, mut wa, mut wc) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut crg, mut mass, mut cg, mut c, mut gm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for b in 0..nb as i32 {
        let s = m.hydrostatics(b);
        vols.extend_from_slice(&s.volumes);
        vol.push(s.volume);
        dm.push(s.displaced_mass);
        cb.extend_from_slice(&s.cb);
        wa.push(s.wp_area);
        wc.extend_from_slice(&s.wp_centroid);
        crg.extend_from_slice(&s.c_rho_g);
        mass.push(s.mass);
        cg.extend_from_slice(&s.cg); // NaN without a centre of gravity
        c.extend_from_slice(&s.c);
        gm.extend_from_slice(&s.metacentric_height); // NaN without the weight
    }
    put(r, QuantityId::HydrostaticsVolumes, vec![nb, 3], &vols);
    put(r, QuantityId::HydrostaticsVolume, vec![nb], &vol);
    put(r, QuantityId::HydrostaticsDisplacedMass, vec![nb], &dm);
    put(r, QuantityId::HydrostaticsCenterOfBuoyancy, vec![nb, 3], &cb);
    put(r, QuantityId::HydrostaticsWaterplaneArea, vec![nb], &wa);
    put(r, QuantityId::HydrostaticsWaterplaneCentroid, vec![nb, 2], &wc);
    put(r, QuantityId::HydrostaticsStiffnessRhoG, vec![nb, 6, 6], &crg);
    put(r, QuantityId::HydrostaticsMass, vec![nb], &mass);
    put(r, QuantityId::HydrostaticsCenterOfGravity, vec![nb, 3], &cg);
    put(r, QuantityId::HydrostaticsStiffness, vec![nb, 6, 6], &c);
    put(r, QuantityId::HydrostaticsMetacentricHeight, vec![nb, 2], &gm);
}

/// The points of the field outputs of a model as quantities of results: the sea's (`field_points`,
/// global) and the tanks' (`tanks`, body axes as the tanks give them, with their tank numbers
/// `tanks_ids`), each when the model has some (reference/results).
pub(crate) fn add_field_points(doc: &Document, r: &mut Results) {
    let sea = sea_points(doc);
    if !sea.is_empty() {
        let v: Vec<f64> = sea.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        put(r, QuantityId::FieldPoints, vec![sea.len() as i64, 3], &v);
    }
    let (pts, ids) = tank_points(doc);
    if !pts.is_empty() {
        let v: Vec<f64> = pts.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        put(r, QuantityId::Tanks, vec![pts.len() as i64, 3], &v);
        put(r, QuantityId::TanksIds, vec![ids.len() as i64], &ids);
    }
}

// the field outputs of the solve by owner: the solve computes them at the field points of the
// problem, the sea's then those of every tank of the bodies in order (document_build::field_points);
// the rows of the sea stay in the field_points_* quantities, those of the tanks go to the tanks_*
// quantities (radiation only: the fluid of a tank has no incident or scattered wave, its rows of the
// other field quantities are zero), with the points of both (add_field_points). Without tanks the
// quantities stay as they are (not copied); with tanks each is split into its two parts while it is
// held (the memory estimate counts that copy). An internal error when a field quantity has not the
// rows of those points.
fn split_field_points(doc: &Document, r: &mut Results) -> Result<()> {
    let j = sea_points(doc).len();
    let n = tank_points(doc).0.len();
    const FIELD: [(QuantityId, Option<QuantityId>); 8] = [
        (QuantityId::FieldPointsPressureRadiation, Some(QuantityId::TanksPressureRadiation)),
        (QuantityId::FieldPointsPressureIncident, None),
        (QuantityId::FieldPointsPressureScattering, None),
        (QuantityId::FieldPointsPressureDiffraction, None),
        (QuantityId::FieldPointsVelocityRadiation, Some(QuantityId::TanksVelocityRadiation)),
        (QuantityId::FieldPointsVelocityIncident, None),
        (QuantityId::FieldPointsVelocityScattering, None),
        (QuantityId::FieldPointsVelocityDiffraction, None),
    ];
    for (q, tank) in FIELD {
        let Some(all) = r.quantities.get(&q) else { continue };
        if all.dims.len() < 2 || all.dims[1] != (j + n) as i64 {
            return Err(Error::internal(format!("{} has not the rows of the field points", q.name())));
        }
        if n == 0 {
            continue; // the sea's rows alone: the quantity as it is
        }
        let Some(all) = r.quantities.remove(&q) else { continue };
        // the elements of a row (a point) and of a frequency
        let row = all.dims[2..].iter().product::<i64>() as usize;
        let per = row * (j + n);
        let lead = all.dims[0] as usize;
        let part = |rows: std::ops::Range<usize>| -> Result<Quantity> {
            let (from, to) = (rows.start * row, rows.end * row);
            let Some(QuantityData::ComplexF32(v)) = all.elements.get().map(|d| &**d) else {
                return Err(Error::internal(format!("{} is not of complex64 elements", q.name())));
            };
            let mut part = Vec::with_capacity(lead * (to - from));
            for f in 0..lead {
                part.extend_from_slice(&v[f * per + from..f * per + to]);
            }
            let mut dims = all.dims.clone();
            dims[1] = rows.len() as i64;
            Ok(Quantity::of(dims, QuantityData::ComplexF32(part)))
        };
        if j > 0 {
            r.quantities.insert(q, part(0..j)?);
        }
        if let Some(t) = tank {
            r.quantities.insert(t, part(j..j + n)?);
        }
    }
    add_field_points(doc, r);
    Ok(())
}

// ------------------------------------------------------------------ the solve job

/// A solve taken out of its model (Model::begin_solve): it owns the validated problem, and may run on
/// another thread; Model::finish_solve takes its outcome. While it lives the model is calculating;
/// dropped, it ends the solve (the state and the cancel request cleared).
pub struct SolveJob {
    problem: Problem,
    built_revision: u64,
    numbering: Numbering, // of the problem in the model

    doc: Arc<Document>,
    calculating: Arc<AtomicBool>,
    cancel: CancelToken,
    board: Arc<Board>,
    hooks: Vec<ActiveHook>, // the hooks of the environment set at the start
}

impl std::fmt::Debug for SolveJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolveJob").field("cancelled", &self.cancel.is_cancelled()).finish()
    }
}

impl SolveJob {
    /// The cancel request of the solve (that of its model).
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }
    /// Runs the solve: the frequencies, the momentum drift (theory §11.1), the hydrostatics with the
    /// results, the coordinates, the meta data (with the hooks of the environment that are set) and
    /// the diagnostics (the warnings of the model before those of the solve, then
    /// Warning::EnvironmentHook when a hook that changes the numbers is set).
    pub fn run(&mut self, progress: Option<&mut ProgressFn<'_>>) -> Result<Results> {
        match progress {
            Some(f) => {
                // the callback from one thread at a time (under the lock); a panic of it cancels the
                // solve and is raised again when the solve returns
                let f = Mutex::new(f);
                let panic: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);
                let cancel = self.cancel.clone();
                let report = |p: &Progress| {
                    let mut g = panic.lock().unwrap_or_else(|e| e.into_inner());
                    if g.is_some() {
                        return;
                    }
                    let mut f = f.lock().unwrap_or_else(|e| e.into_inner());
                    if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (*f)(p))) {
                        *g = Some(e);
                        cancel.cancel();
                    }
                };
                let mut ctx = SolveContext::new(self.cancel.clone(), Some(&report));
                ctx.board = Some(self.board.clone());
                let r = self.run_in(&ctx);
                if let Some(p) = panic.into_inner().unwrap_or_else(|e| e.into_inner()) {
                    std::panic::resume_unwind(p);
                }
                r
            }
            None => {
                let mut ctx = SolveContext::new(self.cancel.clone(), None);
                ctx.board = Some(self.board.clone());
                self.run_in(&ctx)
            }
        }
    }

    // the solve in its context, its end phase reported (done, cancelled or failed)
    fn run_in(&mut self, ctx: &SolveContext) -> Result<Results> {
        let r = self.run_body(ctx);
        ctx.phase(end_phase(&r));
        r
    }

    fn run_body(&mut self, ctx: &SolveContext) -> Result<Results> {
        let t0 = Instant::now();
        let m = &mut self.problem;
        let mut r = solve(m, ctx)?;
        ctx.phase(Phase::Finishing); // the results written
        add_momentum_drift(m, &mut r)?; // theory §11.1
        // Newman's approximation from the momentum drift (a drift asked for: none is computed for
        // it alone)
        crate::solver::newman(m, &mut r, NewmanDrifts { momentum_conservation: true, ..NewmanDrifts::default() });
        add_hydrostatics(m, &mut r); // with the results
        add_coordinates(m, &mut r); // the D axis
        let seconds = t0.elapsed().as_secs_f64();
        r.model = Some(self.doc.clone());
        if m.has_field_points {
            let solved = r.solved_document().ok_or_else(|| Error::internal("the results of a solve hold its model"))?;
            split_field_points(&solved, &mut r)?;
        }
        // the quantities computed only for others (Options::hidden), once every one is in
        crate::solver::drop_hidden(m, &mut r);
        r.meta = results_meta(&r.route, m.opt.frequency_threads, m.opt.inner_threads, &[seconds]);
        r.meta.environment_hooks = self.hooks.iter().map(|h| h.text.clone()).collect();
        // the warnings of the model before those of the solve, located in the model
        let mut diagnostics = m.warnings.clone();
        diagnostics.append(&mut r.diagnostics);
        let changing: Vec<&str> = self.hooks.iter().filter(|h| h.changes_results).map(|h| h.text.as_str()).collect();
        if !changing.is_empty() {
            diagnostics.push(Diagnostic::warning_at(
                Warning::EnvironmentHook,
                Location::default(),
                format!(
                    "the environment sets hooks that change the results (internal studies only; unset them for \
                     the method's results): {}",
                    changing.join(", ")
                ),
            ));
        }
        r.diagnostics = diagnostics.iter().map(|d| self.numbering.diagnostic(d)).collect();
        Ok(*r)
    }
}

impl Drop for SolveJob {
    fn drop(&mut self) {
        // the end of a solve, a failed one too: the state (calculating) and the cancel request are
        // cleared
        self.cancel.clear();
        self.calculating.store(false, Ordering::SeqCst);
        // a job dropped without finish_solve (a panic of the solve, a client that gave up): its
        // frequencies not ended are cancelled (finish_solve ended the board already otherwise)
        self.board.end(Phase::Cancelled);
    }
}
