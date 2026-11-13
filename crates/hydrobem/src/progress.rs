//! The progress of a solve and its cancel request inside the steps: a SolveContext per
//! solve (the cancel token, the progress reporter and the board), passed explicitly as a `Work`
//! view (the context, the frequency of the work and its worker) to the steps that report a stage
//! or check for cancel: the statics, the assembly tiles, the LU panels and the output steps (the
//! field points within theirs, on the board), and the items of the second-order steps
//! (`StepWork`, `ItemWork`). The reporter calls the client's callback (throttled; the phases and
//! the end of every frequency always); the board keeps the state of the solve in atomics for a
//! `ProgressMonitor` to read at any instant: its phase, every frequency, every frequency worker
//! and the steps of the second order with their items.
//!
//! The workers: a solve runs its frequencies on frequency_threads workers (parallel.rs), each with
//! its fixed number 0..f−1 for the whole solve; the second-order steps that run by item (the rows
//! of the quadratic QTFs, the pair frequencies of the potential QTF, the frequencies of its
//! sub-problems) run on the same workers, so that a worker number means the same thread in every
//! phase. The work that runs on all threads at once (the statics, the fits and the outer integral
//! of the potential QTF) is shared by every worker.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use crate::codes::Stage;
use crate::error::{Error, ErrorKind, Result};
use crate::model_types::{Progress, ProgressAt};

/// The cancel request of a model's solve: clone it to another thread and call `cancel`;
/// checked before every static row, assembly tile, LU panel, output step and second-order item
/// (and inside the long ones). Shared by the threads of the solve (its SolveContext) and by the
/// thread that cancels it.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token of no model (`Model::cancel_token` gives that of a model).
    pub fn new() -> CancelToken {
        CancelToken::default()
    }
    /// Requests the cancellation of the running solve, or of the next one when none runs.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    /// Withdraws the request (the end of a solve).
    pub(crate) fn clear(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

// ------------------------------------------------------------------ the names of the progress

/// The phase of a solve, in their order: the frequency-independent data, the limits of the
/// radiation problem, the ordinary frequencies, the second-order loads (only when every
/// frequency is done), the writing of the results, and its end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    Preparation,
    Limits,
    FirstOrder,
    SecondOrder,
    Finishing,
    Done,
    Cancelled,
    Failed,
}

impl Phase {
    const ALL: [Phase; 8] = [
        Phase::Preparation,
        Phase::Limits,
        Phase::FirstOrder,
        Phase::SecondOrder,
        Phase::Finishing,
        Phase::Done,
        Phase::Cancelled,
        Phase::Failed,
    ];
    /// Its name, e.g. "first_order".
    pub fn name(self) -> &'static str {
        match self {
            Phase::Preparation => "preparation",
            Phase::Limits => "limits",
            Phase::FirstOrder => "first_order",
            Phase::SecondOrder => "second_order",
            Phase::Finishing => "finishing",
            Phase::Done => "done",
            Phase::Cancelled => "cancelled",
            Phase::Failed => "failed",
        }
    }
    /// The end of a solve: done, cancelled or failed.
    pub fn is_end(self) -> bool {
        self >= Phase::Done
    }
    fn of(c: u8) -> Phase {
        Phase::ALL[usize::from(c).min(7)]
    }
}

/// A step of the second-order phase, in the order they run: the quadratic QTFs by pressure
/// integration and over the control surfaces (their items the rows ω_a of the pairs), and the
/// potential QTF: its first-order sub-problem (the model's frequencies with the free-surface
/// points; its items those frequencies), its radiation at the pair frequencies (items the pair
/// frequencies), the far-field fits (one item), the outer integral (items its batches of radii),
/// the inner free surface and the body condition of every pair (items the pair frequencies) and
/// the direct method's solve (items the pair frequencies).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SecondOrderStep {
    QuadraticPressure,
    QuadraticControlSurface,
    PotentialFirstOrder,
    PotentialRadiation,
    PotentialFits,
    PotentialOuter,
    PotentialPairs,
    PotentialDirect,
}

impl SecondOrderStep {
    pub const ALL: [SecondOrderStep; 8] = [
        SecondOrderStep::QuadraticPressure,
        SecondOrderStep::QuadraticControlSurface,
        SecondOrderStep::PotentialFirstOrder,
        SecondOrderStep::PotentialRadiation,
        SecondOrderStep::PotentialFits,
        SecondOrderStep::PotentialOuter,
        SecondOrderStep::PotentialPairs,
        SecondOrderStep::PotentialDirect,
    ];
    /// Its name, e.g. "potential_pairs".
    pub fn name(self) -> &'static str {
        match self {
            SecondOrderStep::QuadraticPressure => "quadratic_pressure_integration",
            SecondOrderStep::QuadraticControlSurface => "quadratic_control_surface",
            SecondOrderStep::PotentialFirstOrder => "potential_first_order",
            SecondOrderStep::PotentialRadiation => "potential_radiation",
            SecondOrderStep::PotentialFits => "potential_fits",
            SecondOrderStep::PotentialOuter => "potential_outer",
            SecondOrderStep::PotentialPairs => "potential_pairs",
            SecondOrderStep::PotentialDirect => "potential_direct",
        }
    }
    /// Its index in ALL (the C ABI's HB_STEP_ code).
    pub fn code(self) -> i32 {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0) as i32
    }
}

/// The state of a frequency worker: on an item (or on the work all threads share: the statics,
/// the fits and the outer integral of the potential QTF), idle (no item left for it in the phase
/// while others still run, and while the results are written) or done (the solve has ended).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkerState {
    Running,
    Idle,
    Done,
}

impl WorkerState {
    /// Its name, e.g. "idle".
    pub fn name(self) -> &'static str {
        match self {
            WorkerState::Running => "running",
            WorkerState::Idle => "idle",
            WorkerState::Done => "done",
        }
    }
    fn of(c: u8) -> WorkerState {
        match c {
            0 => WorkerState::Running,
            1 => WorkerState::Idle,
            _ => WorkerState::Done,
        }
    }
}

/// What a worker is on: a limit (its index among the limits), an ordinary frequency, or an item of
/// a second-order step (none: the work of the step all threads share).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Task {
    Limit(usize),
    Frequency(usize),
    SecondOrder { step: SecondOrderStep, item: Option<usize> },
}

// ------------------------------------------------------------------ the callback

/// The callback that receives the reports of a solve, from one thread at a time (Model::solve
/// wraps its FnMut callback in a lock to give one).
pub type ReportFn<'a> = dyn Fn(&Progress) + Sync + 'a;

/// The progress callback of one solve: the stage reports at most every INTERVAL_US and never make
/// a thread wait (a report is skipped while another is being delivered); the end of every
/// frequency and every change of the phase are always reported.
pub struct Reporter<'a> {
    callback: Option<&'a ReportFn<'a>>,
    start: Instant,
    last_us: AtomicI64,
    frequencies_done: AtomicI32,
    frequencies_total: AtomicI32,
    phase: AtomicU8,
    mu: Mutex<()>,
}

impl<'a> Reporter<'a> {
    pub const INTERVAL_US: i64 = 100_000;

    pub fn new(callback: Option<&'a ReportFn<'a>>) -> Reporter<'a> {
        Reporter {
            callback,
            start: Instant::now(),
            last_us: AtomicI64::new(-1_000_000),
            frequencies_done: AtomicI32::new(0),
            frequencies_total: AtomicI32::new(0),
            phase: AtomicU8::new(Phase::Preparation as u8),
            mu: Mutex::new(()),
        }
    }

    pub fn set_frequencies(&self, total: i32) {
        self.frequencies_total.store(total, Ordering::Relaxed);
    }

    fn now_us(&self) -> i64 {
        self.start.elapsed().as_micros() as i64
    }

    /// A stage of the reporting thread: done of total steps (static rows, tiles, LU panels, the
    /// parts of a second-order item), of the work `at` on `worker`.
    pub fn stage(&self, stage: Stage, at: ProgressAt, done: i32, total: i32, worker: Option<usize>) {
        if self.callback.is_none() {
            return;
        }
        let t = self.now_us();
        if t - self.last_us.load(Ordering::Relaxed) < Self::INTERVAL_US {
            return;
        }
        let Ok(_lock) = self.mu.try_lock() else { return }; // another thread is delivering: skip
        self.last_us.store(t, Ordering::Relaxed);
        self.deliver(stage, at, done, total, worker);
    }

    /// One more frequency (or limit) finished on `worker`: always delivered.
    pub fn frequency_done(&self, at: ProgressAt, worker: Option<usize>) {
        if self.callback.is_none() {
            self.frequencies_done.fetch_add(1, Ordering::SeqCst);
            return;
        }
        // counted under the lock, so that the reports of two threads arrive in the order of their counts
        let _lock = self.mu.lock().unwrap_or_else(|e| e.into_inner());
        let d = self.frequencies_done.fetch_add(1, Ordering::SeqCst) + 1;
        self.last_us.store(self.now_us(), Ordering::Relaxed);
        self.deliver(Stage::FrequencyDone, at, d, self.frequencies_total.load(Ordering::Relaxed), worker);
    }

    /// The phase of the solve moves on to p (never back; an end is final): delivered when it
    /// changed. True when it changed.
    pub fn phase(&self, p: Phase) -> bool {
        let mut cur = self.phase.load(Ordering::SeqCst);
        loop {
            let c = Phase::of(cur);
            if c.is_end() || p <= c {
                return false;
            }
            match self.phase.compare_exchange(cur, p as u8, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(x) => cur = x,
            }
        }
        if self.callback.is_some() {
            let _lock = self.mu.lock().unwrap_or_else(|e| e.into_inner());
            self.last_us.store(self.now_us(), Ordering::Relaxed);
            let d = self.frequencies_done.load(Ordering::SeqCst);
            self.deliver(Stage::Phase, ProgressAt::Solve, d, self.frequencies_total.load(Ordering::Relaxed), None);
        }
        true
    }

    fn deliver(&self, stage: Stage, at: ProgressAt, done: i32, total: i32, worker: Option<usize>) {
        let p = Progress::new(
            stage,
            at,
            done,
            total,
            self.frequencies_done.load(Ordering::SeqCst),
            self.frequencies_total.load(Ordering::Relaxed),
            self.now_us() as f64 * 1e-6,
            worker,
            Phase::of(self.phase.load(Ordering::SeqCst)),
        );
        if let Some(f) = self.callback {
            f(&p);
        }
    }
}

// ------------------------------------------------------------------ the context of a solve

/// The context of one solve: its cancel request, its progress reporter and the board of its
/// progress (none for a sub-problem: a tank). A sub-problem of the potential QTF has a board of
/// its own (that of its step on the model's board) and reports its stages through the model's
/// reporter (`forward`), as items of its step.
pub struct SolveContext<'a> {
    pub cancel: CancelToken,
    pub reporter: Reporter<'a>,
    pub(crate) board: Option<Arc<Board>>,
    forward: Option<(&'a Reporter<'a>, SecondOrderStep)>,
}

impl Default for SolveContext<'_> {
    /// No reports, a cancel token of its own.
    fn default() -> Self {
        SolveContext::new(CancelToken::new(), None)
    }
}

impl<'a> SolveContext<'a> {
    pub fn new(cancel: CancelToken, progress: Option<&'a ReportFn<'a>>) -> SolveContext<'a> {
        SolveContext { cancel, reporter: Reporter::new(progress), board: None, forward: None }
    }

    /// The work of the statics, of the limit of an index (among the limits of the solve) or of an
    /// ordinary frequency.
    pub fn work_at(&'a self, at: ProgressAt, limit: usize) -> Work<'a> {
        let slot = self.board.as_ref().and_then(|b| b.slot_of(at, limit));
        Work { ctx: self, at, slot, worker: None }
    }

    /// The work of the statics or of an ordinary frequency (a limit: `work_at` with its index).
    pub fn work(&'a self, at: ProgressAt) -> Work<'a> {
        debug_assert!(at != ProgressAt::Limit, "the work of a limit names its index (work_at)");
        self.work_at(at, 0)
    }

    /// The work of the statics (no frequency).
    pub fn statics(&'a self) -> Work<'a> {
        self.work(ProgressAt::Statics)
    }

    /// ErrorKind::Cancelled when the solve was cancelled.
    pub fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_cancelled() {
            return Err(Error::cancelled("the solve was cancelled"));
        }
        Ok(())
    }

    /// The solve moves on to phase p (never back): the board's workers idle for the new phase, the
    /// callback told (always). A sub-problem's context moves its own board alone.
    pub fn phase(&self, p: Phase) {
        if let Some(b) = &self.board {
            b.set_phase(p);
        }
        if self.forward.is_none() {
            self.reporter.phase(p);
        }
    }

    /// One more frequency (or limit) finished on `worker` (the reports of the model's solve; a
    /// sub-problem's frequencies are items of its step).
    pub fn frequency_done(&self, at: ProgressAt, worker: Option<usize>) {
        if self.forward.is_none() {
            self.reporter.frequency_done(at, worker);
        }
    }

    /// Worker w has no item left in the loop it ran (it waits for the others to end it).
    pub fn worker_idle(&self, w: usize) {
        if let Some(b) = &self.board {
            b.idle(w);
        }
        self.report_to(Stage::WorkerIdle, ProgressAt::Solve, 0, 0, Some(w));
    }

    fn report_to(&self, stage: Stage, at: ProgressAt, done: i32, total: i32, worker: Option<usize>) {
        match self.forward {
            Some((r, step)) => {
                let at = match at {
                    ProgressAt::Solve => ProgressAt::Solve,
                    ProgressAt::SecondOrder { .. } => at,
                    _ => ProgressAt::SecondOrder { step, item: None },
                };
                r.stage(stage, at, done, total, worker)
            }
            None => self.reporter.stage(stage, at, done, total, worker),
        }
    }

    /// A second-order step of `items` items begins (on the board, its workers idle until they take
    /// an item).
    pub fn step(&'a self, step: SecondOrderStep, items: usize) -> StepWork<'a> {
        let slot = self.board.as_ref().map(|b| b.begin_step(step, items, None));
        if let Some(b) = &self.board {
            b.all_workers(WorkerState::Idle, 0);
        }
        StepWork { ctx: self, step, slot }
    }

    /// A sub-problem of the second order begins as the step `step`: its context (the cancel
    /// request of this solve; its board that of the step, its limits and its ordinary frequencies
    /// the items; its stages reported through this solve's callback as items of the step). End it
    /// with `SubProblem::end`.
    pub fn sub_problem(&'a self, step: SecondOrderStep, limits: Vec<Limit>, frequencies: usize) -> SubProblem<'a> {
        let (board, slot) = match &self.board {
            Some(b) => {
                let (s, sub) = b.begin_sub_problem(step, limits, frequencies);
                (Some(sub), Some(s))
            }
            None => (None, None),
        };
        if let (Some(b), Some(s)) = (&self.board, &slot) {
            b.all_workers(WorkerState::Running, b.task_code(s.group, None));
        }
        let ctx = SolveContext {
            cancel: self.cancel.clone(),
            reporter: Reporter::new(None),
            board,
            forward: Some((&self.reporter, step)),
        };
        SubProblem { ctx, slot }
    }
}

/// The view of a step on its solve: the context, the frequency of the work and its worker, handed
/// to the steps (and to the items of their parallel loops) that report a stage or check for cancel.
#[derive(Clone, Copy)]
pub struct Work<'a> {
    pub ctx: &'a SolveContext<'a>,
    pub at: ProgressAt,
    // the slot of the frequency on the board (none: the statics, or no board)
    slot: Option<usize>,
    worker: Option<usize>,
}

impl Work<'_> {
    /// This work on frequency worker w.
    pub fn on(self, w: usize) -> Self {
        Work { worker: Some(w), ..self }
    }
    /// The worker of this work (none: the statics, on all threads).
    pub fn worker(&self) -> Option<usize> {
        self.worker
    }
    /// Reports done of total of a stage for the frequency of this work (throttled).
    pub fn report(&self, stage: Stage, done: i32, total: i32) {
        if let Some(b) = &self.ctx.board {
            b.report(self.slot, stage, done, total);
        }
        let at = match (self.ctx.forward, self.slot) {
            (Some((_, step)), item) => ProgressAt::SecondOrder { step, item },
            (None, _) => self.at,
        };
        self.ctx.report_to(stage, at, done, total, self.worker);
    }
    /// The slot of the frequency of this work on the board of the solve.
    pub fn slot(&self) -> Option<usize> {
        self.slot
    }
    /// Progress inside the step of the stage last reported (`report`): done of total parts of
    /// it, for the board alone (a long step, as the field points, moves the fraction meanwhile;
    /// the callback's reports stay those of the steps).
    pub fn sub(&self, done: usize, total: usize) {
        if let (Some(b), Some(s)) = (&self.ctx.board, self.slot) {
            b.sub(s, done, total);
        }
    }

    /// ErrorKind::Cancelled when the solve was cancelled (`CancelToken::cancel`).
    pub fn check_cancel(&self) -> Result<()> {
        self.ctx.check_cancel()
    }
}

/// A second-order step of a solve (`SolveContext::step`): its items begin on a worker
/// (`item`) or are shared by all (`shared`); `end` ends the step.
pub struct StepWork<'a> {
    pub ctx: &'a SolveContext<'a>,
    pub step: SecondOrderStep,
    slot: Option<Arc<Step>>,
}

impl<'a> StepWork<'a> {
    /// Item i begins on worker w (none: on all the workers, which share it).
    pub fn item(&'a self, i: usize, worker: Option<usize>) -> ItemWork<'a> {
        if let Some(s) = &self.slot {
            s.items[i].begin(worker);
            if let Some(b) = &self.ctx.board {
                let code = b.task_code(s.group, Some(i));
                match worker {
                    Some(w) => b.take(w, code),
                    None => b.all_workers(WorkerState::Running, code),
                }
            }
        }
        let it = ItemWork { step: self, i, worker };
        it.report(0, 1);
        it
    }
    /// The step ended: its items not ended are done (ok) or cancelled.
    pub fn end(&self, ok: bool) {
        if let Some(s) = &self.slot {
            s.end(ok);
        }
    }
    /// ErrorKind::Cancelled when the solve was cancelled.
    pub fn check_cancel(&self) -> Result<()> {
        self.ctx.check_cancel()
    }
}

/// An item of a second-order step in progress (`StepWork::item`).
pub struct ItemWork<'a> {
    step: &'a StepWork<'a>,
    i: usize,
    worker: Option<usize>,
}

impl ItemWork<'_> {
    /// Its index among the items of its step.
    pub fn index(&self) -> usize {
        self.i
    }
    /// done of total parts of the item (its fraction on the board; the callback throttled).
    pub fn report(&self, done: usize, total: usize) {
        if let Some(s) = &self.step.slot {
            s.items[self.i].raise_part(done, total);
        }
        let at = ProgressAt::SecondOrder { step: self.step.step, item: Some(self.i) };
        self.step.ctx.report_to(Stage::SecondOrder, at, done as i32, total as i32, self.worker);
    }
    /// ErrorKind::Cancelled when the solve was cancelled.
    pub fn check_cancel(&self) -> Result<()> {
        self.step.ctx.check_cancel()
    }
    /// The item ended: done, or with the error of its work (failed or cancelled).
    pub fn end<T>(&self, r: &Result<T>) {
        if let Some(s) = &self.step.slot {
            match r {
                Ok(_) => s.items[self.i].finish(),
                Err(e) => s.items[self.i].stop(e.kind() == ErrorKind::Cancelled),
            }
        }
    }
}

/// A sub-problem of the potential QTF in progress (`SolveContext::sub_problem`): its context, to
/// solve it in, and its step.
pub struct SubProblem<'a> {
    pub ctx: SolveContext<'a>,
    slot: Option<Arc<Step>>,
}

impl SubProblem<'_> {
    /// The sub-problem's solve ended: its frequencies not ended are done (ok) or cancelled.
    pub fn end(&self, ok: bool) {
        if let Some(s) = &self.slot {
            s.end(ok);
        }
    }
}

// ------------------------------------------------------------------ the progress board

/// The state of a frequency (or a limit, or an item of a second-order step) in a solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrequencyState {
    /// not begun
    Waiting,
    /// being solved
    Running,
    /// solved
    Done,
    /// its solve failed (the error of the solve names it)
    Failed,
    /// not solved: the solve was cancelled or stopped by the failure of another frequency
    Cancelled,
}

impl FrequencyState {
    /// Its name, e.g. "running".
    pub fn name(self) -> &'static str {
        match self {
            FrequencyState::Waiting => "waiting",
            FrequencyState::Running => "running",
            FrequencyState::Done => "done",
            FrequencyState::Failed => "failed",
            FrequencyState::Cancelled => "cancelled",
        }
    }
}

/// A limit of the radiation problem (theory §1.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Limit {
    /// omega = 0
    Zero,
    /// omega = infinity
    Infinite,
}

impl Limit {
    /// Its name: "zero" or "infinite".
    pub fn name(self) -> &'static str {
        match self {
            Limit::Zero => "zero",
            Limit::Infinite => "infinite",
        }
    }
}

/// The progress of one frequency or limit (or an item of a second-order step): its state, the
/// stage it is in (none before its first step and once it has ended), the fraction of its work
/// done, 0 to 1: the stages weighted by their cost in the solve (never decreasing; 1 once done),
/// and the frequency worker that runs it or ran it (none while it waits, and for an item that all
/// the workers share).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyProgress {
    pub state: FrequencyState,
    pub stage: Option<Stage>,
    pub fraction: f64,
    pub worker: Option<usize>,
}

/// The progress of a limit: which limit, and its progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimitProgress {
    pub limit: Limit,
    pub progress: FrequencyProgress,
}

/// The progress of a step of the second order: which step, its state, the fraction of its work
/// done (the mean of its items', with a sub-problem's preparation as one more; never decreasing,
/// 1 once done) and its items in their order.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderProgress {
    pub step: SecondOrderStep,
    pub state: FrequencyState,
    pub fraction: f64,
    pub items: Vec<FrequencyProgress>,
}

/// A frequency worker at one instant: its state, what it is on (none: idle, done, or the
/// preparation), the stage and the fraction done of that item (0 without one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkerProgress {
    pub state: WorkerState,
    pub task: Option<Task>,
    pub stage: Option<Stage>,
    pub fraction: f64,
}

/// The state of the last (or the running) solve of a model at one instant
/// (`ProgressMonitor::snapshot`): whether it runs, its phase (none before the first solve), the
/// seconds since it began (until it ended), the fraction of the preparation done (the static
/// integrals, 0 to 1), the limits in the order they are solved (zero, then infinite), the ordinary
/// frequencies in the order of the results (the index of `Location::frequency`), the frequencies
/// and limits done of all of them, the frequency workers (frequency_threads of them) and the
/// steps of the second order in the order they began.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolveProgress {
    pub running: bool,
    pub phase: Option<Phase>,
    pub seconds: f64,
    pub preparation: f64,
    pub limits: Vec<LimitProgress>,
    pub frequencies: Vec<FrequencyProgress>,
    pub done: usize,
    pub total: usize,
    pub workers: Vec<WorkerProgress>,
    pub second_order: Vec<SecondOrderProgress>,
}

// the fractions as integers of 1e-9, so that an atomic maximum keeps them from decreasing
const SCALE: f64 = 1e9;
// the stages of a frequency in their order, and their weights before the first frequency has
// measured them (assembly, factorization, outputs): a guess, replaced by the measured weights once
// a frequency is done (a semi-submersible with 2046 field points measured 0.28, 0.06, 0.66; a body
// without field points spends most in the assembly and the LU)
const STAGES: [Stage; 3] = [Stage::Assembly, Stage::Lu, Stage::Outputs];
const FIRST_WEIGHTS: [f64; 3] = [0.45, 0.45, 0.10];
// the stage code of the item of a second-order step (STAGES are 1..=3)
const ITEM_STAGE: u8 = 4;

fn stage_index(s: Stage) -> Option<usize> {
    STAGES.iter().position(|x| *x == s)
}

fn stage_of(c: u8) -> Option<Stage> {
    match c {
        0 => None,
        ITEM_STAGE => Some(Stage::SecondOrder),
        k => Some(STAGES[usize::from(k).min(3) - 1]),
    }
}

fn state_code(s: FrequencyState) -> u8 {
    s as u8
}

fn state_of(c: u8) -> FrequencyState {
    match c {
        0 => FrequencyState::Waiting,
        1 => FrequencyState::Running,
        2 => FrequencyState::Done,
        3 => FrequencyState::Failed,
        _ => FrequencyState::Cancelled,
    }
}

fn raise(a: &AtomicU32, fraction: f64) {
    a.fetch_max((fraction.clamp(0.0, 1.0) * SCALE) as u32, Ordering::Relaxed);
}

fn fraction_of(a: &AtomicU32) -> f64 {
    f64::from(a.load(Ordering::Relaxed)) / SCALE
}

// one frequency, limit or item: its state, its stage (stage_of), its fraction, the weights of its
// stages (fixed when it begins), the microseconds when it began and each stage began, its worker
// (-1 none)
struct Slot {
    state: AtomicU8,
    stage: AtomicU8,
    fraction: AtomicU32,
    // the step in progress of the stage: its start and end as fractions of the stage
    step: [AtomicU32; 2],
    weights: [AtomicU32; 3],
    began_us: AtomicU64,
    stage_us: [AtomicU64; 3],
    worker: AtomicI32,
}

impl Default for Slot {
    fn default() -> Slot {
        Slot {
            state: AtomicU8::new(0),
            stage: AtomicU8::new(0),
            fraction: AtomicU32::new(0),
            step: Default::default(),
            weights: Default::default(),
            began_us: AtomicU64::new(0),
            stage_us: Default::default(),
            worker: AtomicI32::new(-1),
        }
    }
}

impl Slot {
    fn worker(&self) -> Option<usize> {
        usize::try_from(self.worker.load(Ordering::Relaxed)).ok()
    }

    fn progress(&self) -> FrequencyProgress {
        FrequencyProgress {
            state: state_of(self.state.load(Ordering::Acquire)),
            stage: stage_of(self.stage.load(Ordering::Relaxed)),
            fraction: fraction_of(&self.fraction),
            worker: self.worker(),
        }
    }

    // an item of a second-order step begins on a worker (none: shared)
    fn begin(&self, worker: Option<usize>) {
        self.worker.store(worker.map_or(-1, |w| w as i32), Ordering::Relaxed);
        self.stage.store(ITEM_STAGE, Ordering::Relaxed);
        self.state.store(state_code(FrequencyState::Running), Ordering::Release);
    }

    fn raise_part(&self, done: usize, total: usize) {
        let part = if total > 0 { done.min(total) as f64 / total as f64 } else { 1.0 };
        raise(&self.fraction, part);
    }

    fn finish(&self) {
        self.fraction.store(SCALE as u32, Ordering::Relaxed);
        self.stage.store(0, Ordering::Relaxed);
        self.state.store(state_code(FrequencyState::Done), Ordering::Release);
    }

    fn stop(&self, cancelled: bool) {
        let state = if cancelled { FrequencyState::Cancelled } else { FrequencyState::Failed };
        self.stage.store(0, Ordering::Relaxed);
        self.state.store(state_code(state), Ordering::Release);
    }

    // the end of its solve or step: done (succeeded) or cancelled, when it had not ended; true
    // when it is made done
    fn end(&self, succeeded: bool) -> bool {
        let st = state_of(self.state.load(Ordering::Acquire));
        if !matches!(st, FrequencyState::Waiting | FrequencyState::Running) {
            return false;
        }
        self.stage.store(0, Ordering::Relaxed);
        if succeeded {
            self.fraction.store(SCALE as u32, Ordering::Relaxed);
            self.state.store(state_code(FrequencyState::Done), Ordering::Release);
        } else {
            self.state.store(state_code(FrequencyState::Cancelled), Ordering::Release);
        }
        succeeded
    }
}

// a frequency worker: its state (WorkerState) and its task (Board::task_code; 0 none)
struct Worker {
    state: AtomicU8,
    task: AtomicU64,
}

// a step of the second order: its items (a sub-problem's: the slots of its board), its state and
// its group (the number of its items in a worker's task)
pub(crate) struct Step {
    step: SecondOrderStep,
    group: u32,
    state: AtomicU8,
    items: Vec<Slot>,
    sub: Option<Arc<Board>>,
}

impl Step {
    fn end(&self, ok: bool) {
        for s in &self.items {
            s.end(ok);
        }
        if let Some(b) = &self.sub {
            for s in &b.slots {
                s.end(ok);
            }
            if ok {
                raise(&b.preparation, 1.0);
            }
        }
        let st = state_of(self.state.load(Ordering::Acquire));
        if st == FrequencyState::Running {
            let fails = self.items.iter().any(|s| state_of(s.state.load(Ordering::Acquire)) == FrequencyState::Failed);
            let state = if ok {
                FrequencyState::Done
            } else if fails {
                FrequencyState::Failed
            } else {
                FrequencyState::Cancelled
            };
            self.state.store(state_code(state), Ordering::Release);
        }
    }

    fn slots(&self) -> &[Slot] {
        self.sub.as_ref().map_or(&self.items, |b| &b.slots)
    }

    fn progress(&self) -> SecondOrderProgress {
        let state = state_of(self.state.load(Ordering::Acquire));
        let items: Vec<FrequencyProgress> = self.slots().iter().map(Slot::progress).collect();
        let (mut sum, mut n) = (items.iter().map(|i| i.fraction).sum::<f64>(), items.len());
        if let Some(b) = &self.sub {
            sum += fraction_of(&b.preparation);
            n += 1;
        }
        let fraction = if state == FrequencyState::Done {
            1.0
        } else if n > 0 {
            sum / n as f64
        } else {
            0.0
        };
        SecondOrderProgress { step: self.step, state, fraction, items }
    }
}

/// The progress of one solve, written by its threads with atomic stores (no lock, no wait) and
/// read by `ProgressMonitor::snapshot`: the phase, the slots of the limits, then those of the
/// frequencies, the frequency workers and the steps of the second order (a sub-problem of the
/// potential QTF has a board of its own, its workers those of the model's).
pub(crate) struct Board {
    start: Instant,
    limits: Vec<Limit>,
    slots: Vec<Slot>,
    preparation: AtomicU32,
    done: AtomicUsize,
    running: AtomicBool,
    end_us: AtomicU64,
    phase: AtomicU8,
    // the stage weights measured by the frequencies done (a mean), for those that begin later
    measured: Mutex<([f64; 3], usize)>,
    // the number of the board's slots in a worker's task: 0 the model's, k + 1 the step k's
    group: u32,
    workers: Arc<Vec<Worker>>,
    // the steps of the second order (the model's board), in the order they began; written when a
    // step begins, read by the snapshot
    steps: RwLock<Vec<Arc<Step>>>,
}

impl Board {
    fn new(limits: Vec<Limit>, frequencies: usize, group: u32, workers: Arc<Vec<Worker>>) -> Board {
        let n = limits.len() + frequencies;
        Board {
            start: Instant::now(),
            limits,
            slots: (0..n).map(|_| Slot::default()).collect(),
            preparation: AtomicU32::new(0),
            done: AtomicUsize::new(0),
            running: AtomicBool::new(true),
            end_us: AtomicU64::new(0),
            phase: AtomicU8::new(Phase::Preparation as u8),
            measured: Mutex::new((FIRST_WEIGHTS, 0)),
            group,
            workers,
            steps: RwLock::new(Vec::new()),
        }
    }

    fn now_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// The number of slots: the limits and the ordinary frequencies.
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// The slot of a limit (its index among the limits) or of an ordinary frequency.
    pub fn slot_of(&self, at: ProgressAt, limit: usize) -> Option<usize> {
        match at {
            ProgressAt::Limit => Some(limit),
            ProgressAt::Frequency(f) => Some(self.limits.len() + f),
            _ => None,
        }
    }

    // ------------------------------------------------------------ the workers

    // the task of item `item` of group g (none: the work the group's workers share): group << 32 |
    // (item + 1), 0 being no task
    fn task_code(&self, g: u32, item: Option<usize>) -> u64 {
        (u64::from(g) << 32) | item.map_or(0, |i| i as u64 + 1)
    }

    fn set_worker(&self, w: usize, state: WorkerState, task: u64) {
        if let Some(x) = self.workers.get(w) {
            x.task.store(task, Ordering::Relaxed);
            x.state.store(state as u8, Ordering::Release);
        }
    }

    // worker w takes the task
    fn take(&self, w: usize, task: u64) {
        self.set_worker(w, WorkerState::Running, task);
    }

    /// Worker w has no item left (idle).
    pub fn idle(&self, w: usize) {
        self.set_worker(w, WorkerState::Idle, 0);
    }

    fn all_workers(&self, state: WorkerState, task: u64) {
        for w in 0..self.workers.len() {
            self.set_worker(w, state, task);
        }
    }

    /// The phase moves on to p (never back; an end is final): the workers idle for it (done at an
    /// end). True when it changed.
    pub fn set_phase(&self, p: Phase) -> bool {
        let mut cur = self.phase.load(Ordering::SeqCst);
        loop {
            let c = Phase::of(cur);
            if c.is_end() || p <= c {
                return false;
            }
            match self.phase.compare_exchange(cur, p as u8, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(x) => cur = x,
            }
        }
        let state = if p.is_end() { WorkerState::Done } else { WorkerState::Idle };
        self.all_workers(state, 0);
        true
    }

    // ------------------------------------------------------------ the frequencies

    /// A step of a stage: of the preparation (no slot) or of a frequency.
    pub fn report(&self, slot: Option<usize>, stage: Stage, done: i32, total: i32) {
        let part = if total > 0 { f64::from(done.max(0)) / f64::from(total) } else { 1.0 };
        let Some(i) = slot else {
            if stage == Stage::Statics {
                raise(&self.preparation, part);
            }
            return;
        };
        let (Some(s), Some(k)) = (self.slots.get(i), stage_index(stage)) else { return };
        if usize::from(s.stage.load(Ordering::Relaxed)) < k + 1 {
            s.stage.fetch_max(k as u8 + 1, Ordering::Relaxed);
            let _ = s.stage_us[k].compare_exchange(0, self.now_us().max(1), Ordering::Relaxed, Ordering::Relaxed);
        }
        // the step that follows the steps done: done..done + 1 of total
        let next = if total > 0 { f64::from(done.max(0) + 1) / f64::from(total) } else { 1.0 };
        s.step[0].store((part.min(1.0) * SCALE) as u32, Ordering::Relaxed);
        s.step[1].store((next.min(1.0) * SCALE) as u32, Ordering::Relaxed);
        raise(&s.fraction, Self::stage_start(s, k) + Self::weight(s, k) * part);
    }

    fn weight(s: &Slot, k: usize) -> f64 {
        f64::from(f32::from_bits(s.weights[k].load(Ordering::Relaxed)))
    }

    // the fraction of the frequency at the start of its stage k
    fn stage_start(s: &Slot, k: usize) -> f64 {
        (0..k).map(|j| Self::weight(s, j)).sum()
    }

    /// Progress inside the step in progress of the stage of a slot.
    pub fn sub(&self, slot: usize, done: usize, total: usize) {
        let Some(s) = self.slots.get(slot) else { return };
        let stage = usize::from(s.stage.load(Ordering::Relaxed));
        if stage == 0 || stage > 3 || total == 0 {
            return;
        }
        let k = stage - 1;
        let (lo, hi) = (fraction_of(&s.step[0]), fraction_of(&s.step[1]));
        let part = lo + (hi - lo).max(0.0) * (done.min(total) as f64 / total as f64);
        raise(&s.fraction, Self::stage_start(s, k) + Self::weight(s, k) * part);
    }

    /// A frequency begins on a worker (none: no worker known): its stage weights fixed, the
    /// preparation done, the worker on it.
    pub fn begin(&self, slot: usize, worker: Option<usize>) {
        let Some(s) = self.slots.get(slot) else { return };
        let w = self.measured.lock().unwrap_or_else(|p| p.into_inner()).0;
        for (a, v) in s.weights.iter().zip(w) {
            a.store((v as f32).to_bits(), Ordering::Relaxed);
        }
        s.began_us.store(self.now_us().max(1), Ordering::Relaxed);
        s.worker.store(worker.map_or(-1, |w| w as i32), Ordering::Relaxed);
        s.state.store(state_code(FrequencyState::Running), Ordering::Release);
        raise(&self.preparation, 1.0);
        if let Some(w) = worker {
            self.take(w, self.task_code(self.group, Some(slot)));
        }
    }

    /// A frequency is done: its fraction 1, its stage durations measured into the weights.
    pub fn finish_slot(&self, slot: usize) {
        let Some(s) = self.slots.get(slot) else { return };
        let end = self.now_us().max(1);
        // the stages that ran, each until the next began (the last until the end)
        let mut d = [0.0f64; 3];
        let mut t = end;
        for k in (0..3).rev() {
            let b = s.stage_us[k].load(Ordering::Relaxed);
            if b > 0 && b <= t {
                d[k] = (t - b) as f64;
                t = b;
            }
        }
        let sum: f64 = d.iter().sum();
        if sum > 0.0 {
            let mut g = self.measured.lock().unwrap_or_else(|p| p.into_inner());
            let (w, n) = &mut *g;
            for k in 0..3 {
                w[k] = (w[k] * *n as f64 + d[k] / sum) / (*n + 1) as f64;
            }
            *n += 1;
        }
        s.finish();
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// A frequency ended without a result: failed, or cancelled.
    pub fn stop_slot(&self, slot: usize, cancelled: bool) {
        if let Some(s) = self.slots.get(slot) {
            s.stop(cancelled);
        }
    }

    // ------------------------------------------------------------ the second order

    fn push_step(&self, f: impl FnOnce(u32) -> Step) -> Arc<Step> {
        let mut g = self.steps.write().unwrap_or_else(|p| p.into_inner());
        let s = Arc::new(f(g.len() as u32 + 1));
        g.push(s.clone());
        s
    }

    /// A step of n items begins.
    pub fn begin_step(&self, step: SecondOrderStep, n: usize, sub: Option<Arc<Board>>) -> Arc<Step> {
        self.push_step(|group| Step {
            step,
            group,
            state: AtomicU8::new(state_code(FrequencyState::Running)),
            items: (0..n).map(|_| Slot::default()).collect(),
            sub,
        })
    }

    /// A sub-problem of these limits and ordinary frequencies begins as a step: the step and the
    /// board of the sub-problem (its workers the model's).
    pub fn begin_sub_problem(&self, step: SecondOrderStep, limits: Vec<Limit>, n: usize) -> (Arc<Step>, Arc<Board>) {
        let mut board = None;
        let s = self.push_step(|group| {
            let b = Arc::new(Board::new(limits, n, group, self.workers.clone()));
            board = Some(b.clone());
            Step {
                step,
                group,
                state: AtomicU8::new(state_code(FrequencyState::Running)),
                items: Vec::new(),
                sub: Some(b),
            }
        });
        (s, board.expect("the board of the sub-problem"))
    }

    // ------------------------------------------------------------ the end and the snapshot

    /// The end of the solve (once; later calls do nothing) in the phase `end` (done, cancelled or
    /// failed): every slot and step not ended is done (a solve that succeeded) or cancelled (a
    /// solve that failed, was cancelled or was dropped); the workers done.
    pub fn end(&self, end: Phase) {
        if !self.running.load(Ordering::Acquire) {
            return; // ended already
        }
        let succeeded = end == Phase::Done;
        for s in &self.slots {
            if s.end(succeeded) {
                self.done.fetch_add(1, Ordering::Relaxed);
            }
        }
        for st in self.steps.read().unwrap_or_else(|p| p.into_inner()).iter() {
            st.end(succeeded);
        }
        if succeeded {
            raise(&self.preparation, 1.0);
        }
        // the end phase (set_phase refuses a phase after an end, so a failure keeps its own)
        if !self.set_phase(end) {
            self.all_workers(WorkerState::Done, 0);
        }
        self.end_us.store(self.now_us().max(1), Ordering::Relaxed);
        self.running.store(false, Ordering::Release);
    }

    // what worker x is on, with the stage and fraction of that item
    fn worker_progress(&self, x: &Worker, steps: &[Arc<Step>]) -> WorkerProgress {
        let state = WorkerState::of(x.state.load(Ordering::Acquire));
        let code = x.task.load(Ordering::Relaxed);
        let none = WorkerProgress { state, task: None, stage: None, fraction: 0.0 };
        if code == 0 {
            return none;
        }
        let (g, item) = ((code >> 32) as usize, (code & 0xffff_ffff) as usize);
        let item = item.checked_sub(1);
        let nl = self.limits.len();
        let (task, slot) = if g == 0 {
            let Some(i) = item else { return none };
            let task = if i < nl { Task::Limit(i) } else { Task::Frequency(i - nl) };
            (task, self.slots.get(i))
        } else {
            let Some(st) = steps.get(g - 1) else { return none };
            (Task::SecondOrder { step: st.step, item }, item.and_then(|i| st.slots().get(i)))
        };
        let p = slot.map(Slot::progress);
        WorkerProgress {
            state,
            task: Some(task),
            stage: p.and_then(|p| p.stage),
            fraction: p.map_or(0.0, |p| p.fraction),
        }
    }

    fn snapshot(&self) -> SolveProgress {
        let running = self.running.load(Ordering::Acquire);
        let us = if running { self.now_us() } else { self.end_us.load(Ordering::Relaxed) };
        let nl = self.limits.len();
        let steps = self.steps.read().unwrap_or_else(|p| p.into_inner()).clone();
        SolveProgress {
            running,
            phase: Some(Phase::of(self.phase.load(Ordering::SeqCst))),
            seconds: us as f64 * 1e-6,
            preparation: fraction_of(&self.preparation),
            limits: self
                .limits
                .iter()
                .zip(&self.slots)
                .map(|(l, s)| LimitProgress { limit: *l, progress: s.progress() })
                .collect(),
            frequencies: self.slots[nl..].iter().map(Slot::progress).collect(),
            done: self.done.load(Ordering::Relaxed),
            total: self.slots.len(),
            workers: self.workers.iter().map(|x| self.worker_progress(x, &steps)).collect(),
            second_order: steps.iter().map(|s| s.progress()).collect(),
        }
    }
}

/// The progress of a model's solves, to be read from any thread at any time
/// (`Model::progress_monitor`): `snapshot` gives the state of the running solve, or of the last
/// one (none before the first). Reading it never waits for the solve nor slows it.
#[derive(Clone, Debug, Default)]
pub struct ProgressMonitor(Arc<RwLock<Option<Arc<Board>>>>);

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board").field("slots", &self.slots.len()).finish()
    }
}

impl ProgressMonitor {
    /// A monitor of no model (`Model::progress_monitor` gives that of a model).
    pub fn new() -> ProgressMonitor {
        ProgressMonitor::default()
    }
    /// The progress of the running solve, or of the last one; the default (not running, no
    /// phase, nothing to do) before the first.
    pub fn snapshot(&self) -> SolveProgress {
        let board = self.0.read().unwrap_or_else(|p| p.into_inner()).clone();
        board.map(|b| b.snapshot()).unwrap_or_default()
    }
    /// The board of a new solve of these limits and ordinary frequencies on `workers` frequency
    /// workers (all running the preparation).
    pub(crate) fn begin(&self, limits: Vec<Limit>, frequencies: usize, workers: usize) -> Arc<Board> {
        let workers = (0..workers.max(1))
            .map(|_| Worker { state: AtomicU8::new(WorkerState::Running as u8), task: AtomicU64::new(0) })
            .collect();
        let b = Arc::new(Board::new(limits, frequencies, 0, Arc::new(workers)));
        *self.0.write().unwrap_or_else(|p| p.into_inner()) = Some(b.clone());
        b
    }
}
