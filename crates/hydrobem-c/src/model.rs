//! `HbModel`: an engine model, its life cycle, its rules and its solve.
//!
//! The model sits in a mutex locked for each call, so its functions may be called from any
//! thread. A solve takes the built problem out of the model and runs it with the lock released:
//! meanwhile other threads may read the model and cancel the solve (`hb_model_cancel` never waits),
//! and the model refuses changes, queries and `hb_model_free` (`HB_ERR_STATE`) until the solve ends.

use std::ffi::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard};

use hydrobem_core::{
    CancelToken, Diagnostic, Error, FrequencyProgress, Model, ModelState, Phase, Progress, ProgressAt, ProgressFn,
    ProgressMonitor, SecondOrderStep, Severity, Stage, Task,
};

use crate::results::{HbResults, give_results};
use crate::status::{HbLocation, HbStatus, Out, argument, guard, path, put, writable};
use crate::table::{HbTable, give};

/// An engine model (opaque).
pub struct HbModel {
    inner: Mutex<Model>,
    cancel: CancelToken,
    monitor: ProgressMonitor,
}

/// The model has no results.
pub const HB_STATE_RESET: i32 = 0;
/// A solve of the model runs.
pub const HB_STATE_CALCULATING: i32 = 1;
/// The model holds the results of its last solve.
pub const HB_STATE_CALCULATED: i32 = 2;

/// The stage of a progress report: the static integrals.
pub const HB_STAGE_STATICS: i32 = 0;
/// The assembly of the system of a frequency.
pub const HB_STAGE_ASSEMBLY: i32 = 1;
/// The factorization of the system of a frequency.
pub const HB_STAGE_LU: i32 = 2;
/// The outputs of a frequency.
pub const HB_STAGE_OUTPUTS: i32 = 3;
/// A frequency is done.
pub const HB_STAGE_FREQUENCY_DONE: i32 = 4;
/// The parts of an item of a second-order step (`step` and `item` of the report).
pub const HB_STAGE_SECOND_ORDER: i32 = 5;
/// The phase of the solve changed (`phase` of the report; always reported).
pub const HB_STAGE_PHASE: i32 = 6;
/// A frequency worker has no item left in its phase (`worker` of the report).
pub const HB_STAGE_WORKER_IDLE: i32 = 7;

/// The phase of a solve: the frequency-independent influence data.
pub const HB_PHASE_PREPARATION: i32 = 0;
/// The limits of the radiation problem.
pub const HB_PHASE_LIMITS: i32 = 1;
/// The ordinary frequencies.
pub const HB_PHASE_FIRST_ORDER: i32 = 2;
/// The second-order loads (after every frequency).
pub const HB_PHASE_SECOND_ORDER: i32 = 3;
/// The results written.
pub const HB_PHASE_FINISHING: i32 = 4;
/// The solve has ended with its results.
pub const HB_PHASE_DONE: i32 = 5;
/// The solve was cancelled.
pub const HB_PHASE_CANCELLED: i32 = 6;
/// The solve failed.
pub const HB_PHASE_FAILED: i32 = 7;

/// A step of the second order: the quadratic QTF by pressure integration (items the rows omega_a).
pub const HB_STEP_QUADRATIC_PRESSURE_INTEGRATION: i32 = 0;
/// The quadratic QTF over the control surfaces (items the rows omega_a).
pub const HB_STEP_QUADRATIC_CONTROL_SURFACE: i32 = 1;
/// The first-order sub-problem of the potential QTF (items its frequencies).
pub const HB_STEP_POTENTIAL_FIRST_ORDER: i32 = 2;
/// The radiation at the pair frequencies of the potential QTF (items its frequencies).
pub const HB_STEP_POTENTIAL_RADIATION: i32 = 3;
/// The far-field fits of the potential QTF (one item).
pub const HB_STEP_POTENTIAL_FITS: i32 = 4;
/// The outer integral of the potential QTF (items its batches of radii).
pub const HB_STEP_POTENTIAL_OUTER: i32 = 5;
/// The inner free surface and the body of every pair of the potential QTF (items the pair
/// frequencies).
pub const HB_STEP_POTENTIAL_PAIRS: i32 = 6;
/// The direct method's solve of the potential QTF (items its frequencies).
pub const HB_STEP_POTENTIAL_DIRECT: i32 = 7;

/// A report of a solve: the stage (`HB_STAGE_`), the frequency (zero-based; -1 a limit, -2 the
/// statics, -3 an item of a second-order step, -4 the solve as a whole: a phase, an idle worker),
/// the steps of the stage done and in all, the frequencies done and in all, the seconds since the
/// solve began, the frequency worker that reports (0 to frequency_threads - 1; -1 none: the work
/// all threads share), the phase of the solve (`HB_PHASE_`), and for a second-order item its step
/// (`HB_STEP_`, -1 none) and its index among the step's items (-1 none: the work of the step all
/// workers share).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HbProgress {
    pub stage: i32,
    pub frequency: i64,
    pub stage_done: u64,
    pub stage_total: u64,
    pub frequencies_done: u64,
    pub frequencies_total: u64,
    pub seconds: f64,
    pub worker: i32,
    pub phase: i32,
    pub step: i32,
    pub item: i64,
}

/// The progress callback of a solve: called with a report and the caller's pointer, one call at a
/// time, maybe on another thread than the solve's; returning 0 cancels the solve.
pub type HbProgressFn = Option<unsafe extern "C" fn(progress: *const HbProgress, user: *mut c_void) -> i32>;

impl HbModel {
    pub fn of(m: Model) -> HbModel {
        let cancel = m.cancel_token();
        let monitor = m.progress_monitor();
        HbModel { inner: Mutex::new(m), cancel, monitor }
    }

    fn lock(&self) -> MutexGuard<'_, Model> {
        // a panic of an engine call (a bug) leaves the model as it was
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Runs f on the engine model.
    pub fn with<R>(&self, f: impl FnOnce(&mut Model) -> hydrobem_core::Result<R>) -> Out<R> {
        Ok(f(&mut self.lock())?)
    }
}

/// The model of a handle.
///
/// # Safety
/// `m` is null or a model of this library not yet freed.
pub unsafe fn model<'a>(m: *mut HbModel) -> Out<&'a HbModel> {
    if m.is_null() {
        return argument("model is a null pointer");
    }
    // SAFETY: a model of this library (the caller's contract)
    Ok(unsafe { &*m })
}

/// Hands a model to the caller.
///
/// # Safety
/// `out` is null or a writable pointer.
pub unsafe fn give_model(out: *mut *mut HbModel, m: Model) -> Out<()> {
    // refused before the box is made: nothing leaks
    writable(out, "out")?;
    unsafe { put(out, Box::into_raw(Box::new(HbModel::of(m))), "out") }
}

/// The diagnostics as a table: per diagnostic its severity (0 warning, 1 error), code, message
/// and location (body, mesh, element, field_point, lid, frequency; -1 none).
pub fn diagnostics_table(ds: &[Diagnostic]) -> HbTable {
    let n = ds.len();
    let loc: Vec<HbLocation> = ds.iter().map(|d| HbLocation::of(&d.location)).collect();
    let field = |f: fn(&HbLocation) -> i64| loc.iter().map(f).collect::<Vec<_>>();
    HbTable::new()
        .i32s("severity", &[n], ds.iter().map(|d| i32::from(d.severity == Severity::Error)).collect())
        .i32s("code", &[n], ds.iter().map(|d| d.code).collect())
        .texts("message", &ds.iter().map(|d| d.message.as_str()).collect::<Vec<_>>())
        .i64s("body", &[n], field(|l| l.body))
        .i64s("mesh", &[n], field(|l| l.mesh))
        .i64s("element", &[n], field(|l| l.element))
        .i64s("field_point", &[n], field(|l| l.field_point))
        .i64s("lid", &[n], field(|l| l.lid))
        .i64s("frequency", &[n], field(|l| l.frequency))
}

/// A new empty model.
///
/// # Safety
/// `out` is a writable pointer; the model is freed with `hb_model_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_new(out: *mut *mut HbModel) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        unsafe { give_model(out, Model::new()) }
    })
}

/// A model read from a model file (.hbm), or the inputs of a results file (.hbr).
///
/// # Safety
/// `path` is a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_load(path: *const c_char, out: *mut *mut HbModel) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let p = unsafe { self::path(path, "path") }?;
        unsafe { give_model(out, Model::load(&p)?) }
    })
}

/// A calculated model from a results file (.hbr): its inputs and its results.
///
/// # Safety
/// `path` is a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_load_results(path: *const c_char, out: *mut *mut HbModel) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let p = unsafe { self::path(path, "path") }?;
        unsafe { give_model(out, Model::load_results(&p)?) }
    })
}

/// Writes the model file (.hbm); an existing file is kept unless `overwrite` is nonzero.
///
/// # Safety
/// `model` is a model of this library; `path` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_save(model: *mut HbModel, path: *const c_char, overwrite: i32) -> HbStatus {
    guard(|| {
        let p = unsafe { self::path(path, "path") }?;
        unsafe { self::model(model) }?.with(|m| m.save(&p, overwrite != 0))
    })
}

/// Writes the model with its results as a results file (.hbr).
///
/// # Safety
/// `model` is a model of this library; `path` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_save_results(model: *mut HbModel, path: *const c_char, overwrite: i32) -> HbStatus {
    guard(|| {
        let p = unsafe { self::path(path, "path") }?;
        unsafe { self::model(model) }?.with(|m| m.save_results(&p, overwrite != 0))
    })
}

/// A new model of a copy of the inputs (no results).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_copy(model: *mut HbModel, out: *mut *mut HbModel) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let copy = unsafe { self::model(model) }?.with(|m| Ok(m.clone_inputs()))?;
        unsafe { give_model(out, copy) }
    })
}

/// Frees a model (null: nothing). Refused while its solve runs (`HB_ERR_STATE`, nothing freed).
///
/// # Safety
/// `model` is null or a model of this library not yet freed, and no other call on it runs
/// meanwhile or begins before this one returns (the check of a running solve does not make a
/// concurrent call safe); after `HB_OK` neither it nor its object ids are used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_free(model: *mut HbModel) -> HbStatus {
    guard(|| {
        if model.is_null() {
            return Ok(());
        }
        if unsafe { &*model }.lock().state() == ModelState::Calculating {
            return Err(hydrobem_core::Error::state(
                "the model is calculating: it can be freed once the solve ends (nothing was freed)",
            )
            .into());
        }
        // SAFETY: made by Box::into_raw in give_model
        drop(unsafe { Box::from_raw(model) });
        Ok(())
    })
}

/// The state of the model: `HB_STATE_RESET`, `HB_STATE_CALCULATING` or `HB_STATE_CALCULATED`.
///
/// # Safety
/// `model` is a model of this library; `state` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_state(model: *mut HbModel, state: *mut i32) -> HbStatus {
    guard(|| {
        writable(state, "state")?;
        let s = match unsafe { self::model(model) }?.lock().state() {
            ModelState::Reset => HB_STATE_RESET,
            ModelState::Calculating => HB_STATE_CALCULATING,
            ModelState::Calculated => HB_STATE_CALCULATED,
        };
        unsafe { put(state, s, "state") }
    })
}

/// Requests the cancellation of the running solve, or of the next one; from any thread, never
/// waiting for the model.
///
/// # Safety
/// `model` is a model of this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_cancel(model: *mut HbModel) -> HbStatus {
    guard(|| {
        unsafe { self::model(model) }?.cancel.cancel();
        Ok(())
    })
}

type Rows = (Vec<&'static str>, Vec<&'static str>, Vec<f64>, Vec<i64>);

fn rows<'a>(p: impl Iterator<Item = &'a FrequencyProgress> + Clone) -> Rows {
    (
        p.clone().map(|f| f.state.name()).collect(),
        p.clone().map(|f| f.stage.map_or("", Stage::name)).collect(),
        p.clone().map(|f| f.fraction).collect(),
        p.map(|f| f.worker.map_or(-1, |w| w as i64)).collect(),
    )
}

fn opt_index(i: Option<usize>) -> i64 {
    i.map_or(-1, |i| i as i64)
}

/// The progress of the running solve, or of the last one, at this instant: from any thread,
/// without waiting for the model (a solve may run); before the first solve nothing to do. The
/// table: running (0 or 1), seconds (since the solve began, until it ended), preparation (the
/// frequency-independent influence data done, 0 to 1), done and total (the frequencies and limits
/// done and in all); per limit, in the order they are solved: limit (`zero`, `infinite`),
/// limit_state, limit_stage, limit_fraction, limit_worker; per ordinary frequency, in the order of
/// the results (the `frequency` of an error's location): state (`waiting`, `running`, `done`,
/// `failed`, `cancelled`), stage (`assembly`, `lu`, `outputs`; "" none), fraction (its work done,
/// 0 to 1, the stages weighted by their cost; never decreasing, 1 once done) and worker (the
/// frequency worker that runs or ran it, -1 none). phase (`preparation`, `limits`, `first_order`,
/// `second_order`, `finishing`, `done`, `cancelled`, `failed`; "" before the first solve). Per
/// frequency worker (frequency_threads of them): worker_state (`running`, `idle`, `done`),
/// worker_task (what it is on: `limit`, `frequency`, the name of a second-order step; "" none),
/// worker_index (the limit, the frequency or the step's item; -1 none), worker_stage and
/// worker_fraction (of that item). Per step of the second order, in the order they began:
/// step (its name), step_state, step_fraction and step_items (its number of items); per item of
/// every step, the steps one after the other: item_state, item_stage, item_fraction, item_worker
/// (-1 none: run by all the workers, or not begun).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_progress(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let p = unsafe { self::model(model) }?.monitor.snapshot();
        let names: Vec<&str> = p.limits.iter().map(|l| l.limit.name()).collect();
        let (ls, lg, lf, lw) = rows(p.limits.iter().map(|l| &l.progress));
        let (fs, fg, ff, fw) = rows(p.frequencies.iter());
        let (nl, nf) = (lf.len(), ff.len());
        let task = |t: &Option<Task>| -> (&'static str, i64) {
            match t {
                None => ("", -1),
                Some(Task::Limit(i)) => ("limit", *i as i64),
                Some(Task::Frequency(f)) => ("frequency", *f as i64),
                Some(Task::SecondOrder { step, item }) => (step.name(), opt_index(*item)),
            }
        };
        let nw = p.workers.len();
        let ws: Vec<&str> = p.workers.iter().map(|w| w.state.name()).collect();
        let wt: Vec<&str> = p.workers.iter().map(|w| task(&w.task).0).collect();
        let wi: Vec<i64> = p.workers.iter().map(|w| task(&w.task).1).collect();
        let wg: Vec<&str> = p.workers.iter().map(|w| w.stage.map_or("", Stage::name)).collect();
        let wf: Vec<f64> = p.workers.iter().map(|w| w.fraction).collect();
        let ns = p.second_order.len();
        let sn: Vec<&str> = p.second_order.iter().map(|s| s.step.name()).collect();
        let ss: Vec<&str> = p.second_order.iter().map(|s| s.state.name()).collect();
        let sf: Vec<f64> = p.second_order.iter().map(|s| s.fraction).collect();
        let si: Vec<i64> = p.second_order.iter().map(|s| s.items.len() as i64).collect();
        let (is, ig, ifr, iw) = rows(p.second_order.iter().flat_map(|s| s.items.iter()));
        let ni = ifr.len();
        let t = HbTable::new()
            .bool("running", p.running)
            .texts("phase", &[p.phase.map_or("", Phase::name)])
            .f64("seconds", p.seconds)
            .f64("preparation", p.preparation)
            .i64("done", p.done as i64)
            .i64("total", p.total as i64)
            .texts("limit", &names)
            .texts("limit_state", &ls)
            .texts("limit_stage", &lg)
            .f64s("limit_fraction", &[nl], lf)
            .i64s("limit_worker", &[nl], lw)
            .texts("state", &fs)
            .texts("stage", &fg)
            .f64s("fraction", &[nf], ff)
            .i64s("worker", &[nf], fw)
            .texts("worker_state", &ws)
            .texts("worker_task", &wt)
            .i64s("worker_index", &[nw], wi)
            .texts("worker_stage", &wg)
            .f64s("worker_fraction", &[nw], wf)
            .texts("step", &sn)
            .texts("step_state", &ss)
            .f64s("step_fraction", &[ns], sf)
            .i64s("step_items", &[ns], si)
            .texts("item_state", &is)
            .texts("item_stage", &ig)
            .f64s("item_fraction", &[ni], ifr)
            .i64s("item_worker", &[ni], iw);
        unsafe { give(out, t) }
    })
}

/// Discards the results, the inputs kept.
///
/// # Safety
/// `model` is a model of this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_reset(model: *mut HbModel) -> HbStatus {
    guard(|| unsafe { self::model(model) }?.with(|m| m.reset()))
}

/// Every input and the results back to those of a new model.
///
/// # Safety
/// `model` is a model of this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_clear(model: *mut HbModel) -> HbStatus {
    guard(|| unsafe { self::model(model) }?.with(|m| m.clear()))
}

/// The rules of the model (reference/inputs-rules §1), then the refusals of the solver routes
/// (`HB_ERR_UNSUPPORTED`, the message of `hb_model_route`) and, with the potential QTF, the zones
/// of its free surface; its warnings as a diagnostics table (`out` may be null).
///
/// # Safety
/// `model` is a model of this library; `out` null or a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_check(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        let w = unsafe { self::model(model) }?.with(|m| {
            m.check()?;
            Ok(m.diagnostics())
        })?;
        if out.is_null() { Ok(()) } else { unsafe { give(out, diagnostics_table(&w)) } }
    })
}

/// The warnings of the last rule check as a diagnostics table.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_diagnostics(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let w = unsafe { self::model(model) }?.with(|m| Ok(m.diagnostics()))?;
        unsafe { give(out, diagnostics_table(&w)) }
    })
}

fn stage_code(s: Stage) -> i32 {
    match s {
        Stage::Statics => HB_STAGE_STATICS,
        Stage::Assembly => HB_STAGE_ASSEMBLY,
        Stage::Lu => HB_STAGE_LU,
        Stage::Outputs => HB_STAGE_OUTPUTS,
        Stage::FrequencyDone => HB_STAGE_FREQUENCY_DONE,
        Stage::SecondOrder => HB_STAGE_SECOND_ORDER,
        Stage::Phase => HB_STAGE_PHASE,
        Stage::WorkerIdle => HB_STAGE_WORKER_IDLE,
    }
}

fn step_code(s: SecondOrderStep) -> i32 {
    s.code()
}

fn report(p: &Progress) -> HbProgress {
    HbProgress {
        stage: stage_code(p.stage),
        frequency: match p.at {
            ProgressAt::Statics => -2,
            ProgressAt::Limit => -1,
            ProgressAt::Frequency(f) => f as i64,
            ProgressAt::SecondOrder { .. } => -3,
            ProgressAt::Solve => -4,
        },
        stage_done: p.stage_done as u64,
        stage_total: p.stage_total as u64,
        frequencies_done: p.frequencies_done as u64,
        frequencies_total: p.frequencies_total as u64,
        seconds: p.seconds,
        worker: p.worker.map_or(-1, |w| w as i32),
        phase: p.phase as i32,
        step: match p.at {
            ProgressAt::SecondOrder { step, .. } => step_code(step),
            _ => -1,
        },
        item: match p.at {
            ProgressAt::SecondOrder { item, .. } => opt_index(item),
            _ => -1,
        },
    }
}

// the caller's pointer, handed to the callback on the thread that reports
struct User(*mut c_void);
// SAFETY: the pointer is the caller's; the ABI states that the callback may run on another thread
unsafe impl Send for User {}

/// Solves the model on the calling thread (the model's lock released meanwhile); its results are
/// kept in the model and a copy handed out (`out` may be null). `progress` (may be null) is called
/// with each report; returning 0 cancels the solve (`HB_ERR_CANCELLED`). A solve whose memory
/// estimate exceeds the available physical memory is refused before it starts (`HB_ERR_VALUE`;
/// the option `memory_check`).
///
/// # Safety
/// `model` is a model of this library; `progress` null or a function that may be called from
/// another thread with `user`; `out` null or a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_solve(
    model: *mut HbModel,
    progress: HbProgressFn,
    user: *mut c_void,
    out: *mut *mut HbResults,
) -> HbStatus {
    guard(|| {
        let hm = unsafe { self::model(model) }?;
        let mut job = hm.with(|m| m.begin_solve())?;
        let cancel = &hm.cancel;
        let user = User(user);
        let mut call = move |p: &Progress| {
            let Some(f) = progress else { return };
            let u = &user;
            // SAFETY: the caller's function with the caller's pointer (its contract)
            if unsafe { f(&report(p), u.0) } == 0 {
                cancel.cancel();
            }
        };
        let callback: Option<&mut ProgressFn<'_>> = if progress.is_some() { Some(&mut call) } else { None };
        // a panic of the solve still ends it, so that the model leaves the calculating state
        let outcome = catch_unwind(AssertUnwindSafe(|| job.run(callback)))
            .unwrap_or_else(|_| Err(Error::internal("internal error of the engine in the solve (please report it)")));
        let r = hm.with(|m| m.finish_solve(job, outcome).cloned())?;
        if out.is_null() { Ok(()) } else { unsafe { give_results(out, r) } }
    })
}

/// A copy of the results of the last solve (`HB_ERR_STATE` when the model is not calculated).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_results(model: *mut HbModel, out: *mut *mut HbResults) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::model(model) }?.with(|m| m.results().cloned())?;
        unsafe { give_results(out, r) }
    })
}
