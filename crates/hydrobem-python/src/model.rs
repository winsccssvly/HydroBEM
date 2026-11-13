//! `Model`: an engine model held by Python, its life cycle, its queries and its solve.
//!
//! The engine model sits in a mutex, locked for each call (while the lock is awaited the thread is
//! detached from Python, so that the lock and the GIL cannot deadlock). A solve takes the built
//! problem out of the model as a `SolveJob` (`Model::begin_solve`) and runs it with the thread
//! detached and the lock released: the progress callback (which attaches again) and other threads
//! read the model meanwhile; its state is calculating, and the model refuses changes, queries and
//! `close` until `Model::finish_solve`. The cancel token of the model is kept beside the mutex, so
//! that `cancel` never waits for it.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use hydrobem_core::{
    AnalyticShape, CancelToken, ErrorKind, FrequencyProgress, Hydrostatics, MeshFile, Model, ModelState, ObjectId,
    Progress, ProgressAt, ProgressFn, ProgressMonitor, QuantityData, QuantityId, Results, Stage, Task,
};
use numpy::{PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBool, PyDict};

use crate::convert::{array, panels_dict, quality_dict, rows, tessellation_dict};
use crate::errors::{ToPy, diagnostics};
use crate::object::PyObj;
use crate::results::PyResults;

/// (inertia, stiffness, damping, ready) of `motion_equation`.
type MotionEquation<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, bool);
// the columns of classify_points: the status names and the body, lid and tank of each point
type PointColumns<'py> = (Vec<&'static str>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);

#[pyclass(frozen, name = "Model", module = "hydrobem._core")]
pub struct PyModel {
    inner: Mutex<Option<Model>>,
    cancel: CancelToken,
    monitor: ProgressMonitor,
}

impl PyModel {
    pub fn of(m: Model) -> PyModel {
        let cancel = m.cancel_token();
        let monitor = m.progress_monitor();
        PyModel { inner: Mutex::new(Some(m)), cancel, monitor }
    }

    fn lock(&self, py: Python<'_>) -> MutexGuard<'_, Option<Model>> {
        // a panic of an engine call (a bug, raised as PanicException) leaves the model as it was
        self.inner.lock_py_attached(py).unwrap_or_else(|p| p.into_inner())
    }

    /// Runs f on the engine model; closed: a RuntimeError with the message `closed`.
    pub fn with_or<R>(
        &self,
        py: Python<'_>,
        closed: &str,
        f: impl FnOnce(&mut Model) -> hydrobem_core::Result<R>,
    ) -> PyResult<R> {
        let mut g = self.lock(py);
        let m = g.as_mut().ok_or_else(|| PyRuntimeError::new_err(closed.to_string()))?;
        f(m).py()
    }

    pub fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&mut Model) -> hydrobem_core::Result<R>) -> PyResult<R> {
        self.with_or(py, "the model is closed", f)
    }

    fn mesh(slf: &Bound<'_, Self>, made: PyResult<ObjectId>) -> PyResult<PyObj> {
        Ok(PyObj::new(slf.clone().unbind(), made?))
    }

    /// The id of a body given to a function of the model: an object of this model.
    fn own(slf: &Bound<'_, Self>, body: Option<PyRef<'_, PyObj>>) -> PyResult<Option<ObjectId>> {
        match body {
            None => Ok(None),
            Some(b) if b.model.bind(slf.py()).is(slf) => Ok(Some(b.id)),
            Some(_) => Err(PyValueError::new_err("body: a Body of this model, its name or its index")),
        }
    }
}

/// The hydrostatics of a body (theory §3.2) as a dict of the fields of hb.Hydrostatics.
pub fn hydrostatics_dict<'py>(py: Python<'py>, h: &Hydrostatics) -> PyResult<Bound<'py, PyDict>> {
    let x = PyDict::new(py);
    x.set_item("volumes", h.volumes.to_vec())?;
    x.set_item("volume", h.volume)?;
    x.set_item("displaced_mass", h.displaced_mass)?;
    x.set_item("center_of_buoyancy", h.cb.to_vec())?;
    x.set_item("waterplane_area", h.wp_area)?;
    x.set_item("waterplane_centroid", h.wp_centroid.to_vec())?;
    x.set_item("stiffness_rho_g", array(py, h.c_rho_g.to_vec(), &[6, 6])?)?;
    x.set_item("has_weight", h.has_weight)?;
    x.set_item("mass", h.mass)?;
    x.set_item("center_of_gravity", h.cg.to_vec())?;
    x.set_item("stiffness", array(py, h.c.to_vec(), &[6, 6])?)?;
    x.set_item("metacentric_height", h.metacentric_height.to_vec())?;
    Ok(x)
}

/// (running, seconds, preparation, limits, frequencies, done, total, phase, workers, second_order)
/// of `Model.progress`.
type ProgressTuple = (
    bool,
    f64,
    f64,
    Vec<(&'static str, Row)>,
    Vec<Row>,
    usize,
    usize,
    Option<&'static str>,
    Vec<WorkerRow>,
    Vec<(&'static str, &'static str, f64, Vec<Row>)>,
);
/// (state, stage, fraction, worker) of a frequency, a limit or a second-order item.
type Row = (&'static str, Option<&'static str>, f64, Option<usize>);
/// (state, task, index, stage, fraction) of a frequency worker.
type WorkerRow = (&'static str, Option<&'static str>, Option<usize>, Option<&'static str>, f64);

fn progress_row(p: &FrequencyProgress) -> Row {
    (p.state.name(), p.stage.map(Stage::name), p.fraction, p.worker)
}

/// The report of a solve as the tuple the package's callback takes: (stage, frequency (None for the
/// statics, a second-order item and the solve as a whole, -1 a limit), stage_done, stage_total,
/// frequencies_done, frequencies_total, seconds, worker, phase, step, item).
type Report = (
    &'static str,
    Option<i64>,
    usize,
    usize,
    usize,
    usize,
    f64,
    Option<usize>,
    &'static str,
    Option<&'static str>,
    Option<usize>,
);

fn report(p: &Progress) -> Report {
    let (frequency, step, item) = match p.at {
        ProgressAt::Statics | ProgressAt::Solve => (None, None, None),
        ProgressAt::Limit => (Some(-1), None, None),
        ProgressAt::Frequency(f) => (Some(f as i64), None, None),
        ProgressAt::SecondOrder { step, item } => (None, Some(step.name()), item),
    };
    (
        p.stage.name(),
        frequency,
        p.stage_done,
        p.stage_total,
        p.frequencies_done,
        p.frequencies_total,
        p.seconds,
        p.worker,
        p.phase.name(),
        step,
        item,
    )
}

/// The elements of an array of the element type of quantity q (converted by the package).
fn quantity_data(q: QuantityId, a: &Bound<'_, PyUntypedArray>) -> PyResult<QuantityData> {
    fn elements<T: numpy::Element + Copy>(a: &Bound<'_, PyUntypedArray>) -> PyResult<Vec<T>> {
        let typed: PyReadonlyArrayDyn<'_, T> = a.extract()?;
        Ok(typed.as_array().iter().copied().collect())
    }
    use hydrobem_core::ElementType as E;
    Ok(match q.element_type() {
        E::Float64 => QuantityData::Float64(elements(a)?),
        E::Float32 => QuantityData::Float32(elements(a)?),
        E::ComplexF32 => QuantityData::ComplexF32(elements(a)?),
        E::Int32 => QuantityData::Int32(elements(a)?),
        E::ComplexF64 => QuantityData::ComplexF64(elements(a)?),
        E::Int64 => QuantityData::Int64(elements(a)?),
        E::Text => return Err(PyValueError::new_err(format!("{} holds text", q.name()))),
    })
}

#[pymethods]
impl PyModel {
    /// An empty model.
    #[new]
    fn new() -> PyModel {
        PyModel::of(Model::new())
    }

    /// A model file (.hbm), or the inputs of a results file (.hbr).
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<PyModel> {
        py.detach(|| Model::load(&path)).py().map(PyModel::of)
    }

    /// A calculated model from a results file: its inputs and its results.
    #[staticmethod]
    fn load_results(py: Python<'_>, path: PathBuf) -> PyResult<PyModel> {
        py.detach(|| Model::load_results(&path)).py().map(PyModel::of)
    }

    /// Writes the model file (.hbm).
    #[pyo3(signature = (path, overwrite = false))]
    fn save(&self, py: Python<'_>, path: PathBuf, overwrite: bool) -> PyResult<()> {
        self.with(py, |m| m.save(&path, overwrite))
    }

    /// Writes the model with its results as a results file (.hbr).
    #[pyo3(signature = (path, overwrite = false))]
    fn save_results(&self, py: Python<'_>, path: PathBuf, overwrite: bool) -> PyResult<()> {
        self.with(py, |m| m.save_results(&path, overwrite))
    }

    /// An independent copy of the inputs (no results).
    fn copy(&self, py: Python<'_>) -> PyResult<PyModel> {
        self.with(py, |m| Ok(m.clone_inputs())).map(PyModel::of)
    }

    /// Frees the model now; refused while its solve runs.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut g = self.lock(py);
        if g.as_ref().is_some_and(|m| m.state() == ModelState::Calculating) {
            return Err(PyRuntimeError::new_err(
                "the model is calculating: it can be closed once the solve ends (nothing was freed)",
            ));
        }
        let gone = g.take();
        drop(g);
        drop(gone);
        Ok(())
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> bool {
        self.lock(py).is_none()
    }

    /// "reset", "calculating" or "calculated".
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<&'static str> {
        self.with(py, |m| {
            Ok(match m.state() {
                ModelState::Reset => "reset",
                ModelState::Calculating => "calculating",
                ModelState::Calculated => "calculated",
            })
        })
    }

    /// Requests the cancellation of the running solve, or of the next one (any thread).
    fn cancel(&self) {
        self.cancel.cancel();
    }

    /// The progress of the running solve, or of the last one, without waiting for the model:
    /// (running, seconds, preparation, limits as (limit, (state, stage, fraction, worker)),
    /// frequencies as (state, stage, fraction, worker), done, total, phase, workers as (state,
    /// task, index, stage, fraction), second-order steps as (step, state, fraction, items)); the
    /// stage None between the stages.
    fn progress(&self) -> ProgressTuple {
        let p = self.monitor.snapshot();
        let limits = p.limits.iter().map(|l| (l.limit.name(), progress_row(&l.progress))).collect();
        let frequencies = p.frequencies.iter().map(progress_row).collect();
        let workers = p
            .workers
            .iter()
            .map(|w| {
                let (task, index) = match w.task {
                    None => (None, None),
                    Some(Task::Limit(i)) => (Some("limit"), Some(i)),
                    Some(Task::Frequency(f)) => (Some("frequency"), Some(f)),
                    Some(Task::SecondOrder { step, item }) => (Some(step.name()), item),
                };
                (w.state.name(), task, index, w.stage.map(Stage::name), w.fraction)
            })
            .collect();
        let steps = p
            .second_order
            .iter()
            .map(|s| (s.step.name(), s.state.name(), s.fraction, s.items.iter().map(progress_row).collect()))
            .collect();
        (
            p.running,
            p.seconds,
            p.preparation,
            limits,
            frequencies,
            p.done,
            p.total,
            p.phase.map(|x| x.name()),
            workers,
            steps,
        )
    }

    /// The rules of the model; the warnings (hb.Diagnostic).
    fn check<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let w = self.with(py, |m| {
            m.check()?;
            Ok(m.diagnostics().to_vec())
        })?;
        diagnostics(py, &w)
    }

    /// The warnings of the last rule check.
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let w = self.with(py, |m| Ok(m.diagnostics().to_vec()))?;
        diagnostics(py, &w)
    }

    /// The object of a path ("" the model, "options", "bodies/hull", "bodies/hull/tanks/1").
    #[pyo3(signature = (path = ""))]
    fn object(slf: &Bound<'_, Self>, path: &str) -> PyResult<PyObj> {
        let id = slf.get().with(slf.py(), |m| m.object(path))?;
        Ok(PyObj::new(slf.clone().unbind(), id))
    }

    // ---- the queries (after the model rules)

    fn route(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |m| m.route().map(|r| r.name()))
    }

    /// The sizes of the problem: the fields of hb.Counts, per_body as dicts of hb.BodyCounts.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (c, bodies, lids, modes) = self.with(py, |m| {
            let c = m.counts()?;
            Ok((c, m.solved_body_counts()?, m.lid_elements()?, m.radiation_mode_ids()?))
        })?;
        let x = PyDict::new(py);
        x.set_item("bodies", c.bodies)?;
        x.set_item("meshes", c.meshes)?;
        x.set_item("lids", c.lids)?;
        x.set_item("ordinary_frequencies", c.ordinary_frequencies)?;
        x.set_item("headings", c.headings)?;
        x.set_item("coordinates", c.coordinates)?;
        x.set_item("radiation_columns", c.radiation_columns)?;
        x.set_item("field_points", c.field_points)?;
        x.set_item("tank_points", c.tank_points)?;
        x.set_item("lid_elements", c.lid_elements)?;
        x.set_item("pressure_elements", c.pressure_elements)?;
        x.set_item("has_zero_frequency", c.has_zero_frequency)?;
        x.set_item("has_infinite_frequency", c.has_infinite_frequency)?;
        x.set_item("radiation_mode_ids", modes)?;
        x.set_item("per_lid_elements", lids)?;
        let mut per_body = Vec::new();
        for b in bodies {
            let d = PyDict::new(py);
            d.set_item("stored", b.stored)?;
            d.set_item("wetted", b.wetted)?;
            d.set_item("wetted_with_reflections", b.wetted_with_reflections)?;
            d.set_item("dipole", b.dipole)?;
            d.set_item("interior_surface", b.interior_surface.map_or(-1, |n| n as i64))?;
            d.set_item("pressure_modes", b.pressure_modes)?;
            d.set_item("symmetry_copies", b.symmetry_copies)?;
            d.set_item("generalized_modes", b.generalized_modes)?;
            per_body.push(d);
        }
        x.set_item("per_body", per_body)?;
        Ok(x)
    }

    /// (inertia, stiffness, damping, ready): ndof x ndof arrays of the motion equation (theory §3.1).
    fn motion_equation<'py>(&self, py: Python<'py>) -> PyResult<MotionEquation<'py>> {
        let e = self.with(py, |m| m.motion_equation())?;
        let n = [e.ndof, e.ndof];
        Ok((array(py, e.inertia, &n)?, array(py, e.stiffness, &n)?, array(py, e.damping, &n)?, e.ready))
    }

    /// The elements as the engine uses them (after a trim), of every body and of no body, of the
    /// roles named (none: the default roles): a dict of arrays.
    fn panels<'py>(
        &self,
        py: Python<'py>,
        include_reflections: bool,
        roles: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let roles: Vec<&str> = roles.iter().map(String::as_str).collect();
        panels_dict(py, &self.with(py, |m| m.panels(&roles, include_reflections))?)
    }

    /// The points (n, 3) and weights (n,) of one zone of the potential QTF's free surface.
    fn free_surface_points<'py>(&self, py: Python<'py>, zone: &str) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let p = self.with(py, |m| m.free_surface_points(zone))?;
        let flat: Vec<f64> = p.points.iter().flatten().copied().collect();
        let pts = numpy::PyArray1::from_vec(py, flat).reshape([p.points.len(), 3])?;
        Ok((pts.into_any().unbind(), numpy::PyArray1::from_vec(py, p.weights).into_any().unbind()))
    }

    /// The tessellation of the elements of `panels` (the same roles): a dict of arrays.
    fn tessellation<'py>(
        &self,
        py: Python<'py>,
        roles: Vec<String>,
        divisions: [i32; 2],
        include_reflections: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let roles: Vec<&str> = roles.iter().map(String::as_str).collect();
        tessellation_dict(py, &self.with(py, |m| m.tessellation(&roles, divisions, include_reflections))?)
    }

    /// The quality of the meshes of every body: a dict of arrays.
    fn mesh_quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        quality_dict(py, &self.with(py, |m| m.mesh_quality())?)
    }

    /// The displacements of every coordinate at points of the bodies: [n, D, 3].
    fn mode_displacements<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        bodies: Vec<i32>,
        elements: Vec<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pts = rows::<3>(&points, "points")?;
        let n = pts.len();
        let (v, d) = self.with(py, |m| {
            let v = m.mode_displacements(&pts, &bodies, &elements)?;
            Ok((v, m.counts()?.coordinates))
        })?;
        array(py, v, &[n, d, 3])
    }

    /// The memory of a solve, in bytes: statics, per_frequency, outputs, peak (resident), commit.
    fn memory_estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = self.with(py, |m| m.memory_estimate())?;
        let x = PyDict::new(py);
        x.set_item("statics", e.statics)?;
        x.set_item("per_frequency", e.per_frequency)?;
        x.set_item("outputs", e.outputs)?;
        x.set_item("peak", e.peak)?;
        x.set_item("commit", e.commit)?;
        Ok(x)
    }

    /// The verdict of the field-point rules on points (reference/inputs-model-objects §3), points in the sea (tank None)
    /// or in a tank (its object; body coordinates): (status names, the body, the lid and the tank
    /// of each, else -1).
    #[pyo3(signature = (points, tank = None))]
    fn classify_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        tank: Option<Bound<'py, PyObj>>,
    ) -> PyResult<PointColumns<'py>> {
        let pts = rows::<3>(&points, "points")?;
        let tank = tank.map(|t| t.get().id);
        let v = self.with(py, |m| m.classify_points(&pts, tank))?;
        let status = v.iter().map(|c| c.status.name()).collect();
        let n = v.len();
        let col = |f: fn(&hydrobem_core::PointClass) -> Option<usize>| {
            array(py, v.iter().map(|c| f(c).map_or(-1, |i| i as i32)).collect(), &[n])
        };
        Ok((status, col(|c| c.body)?, col(|c| c.lid)?, col(|c| c.tank)?))
    }

    /// Results of the present inputs from arrays by quantity name, each converted to the
    /// quantity's element type by the caller.
    fn new_results(&self, py: Python<'_>, quantities: Vec<(String, Bound<'_, PyUntypedArray>)>) -> PyResult<PyResults> {
        let mut given = Vec::new();
        for (name, a) in &quantities {
            let q = QuantityId::from_name(name)
                .ok_or_else(|| PyValueError::new_err(format!("no result quantity {name:?}")))?;
            given.push((q, a.shape().to_vec(), quantity_data(q, a)?));
        }
        let r = self.with(py, |m| {
            let mut r = m.new_results()?;
            for (q, dims, data) in given {
                r.put(q, &dims, data)?;
            }
            // the quantities given together, once all are put
            r.check_given()?;
            Ok(r)
        })?;
        Ok(PyResults::of(r))
    }

    /// Solves the model with the thread detached from Python; its results are kept in it
    /// and a copy returned. progress(report) is called with the tuple of `report` for each report;
    /// returning False cancels the solve; an exception raised in it cancels the solve and is raised
    /// here.
    #[pyo3(signature = (progress = None))]
    fn solve(&self, py: Python<'_>, progress: Option<Py<PyAny>>) -> PyResult<PyResults> {
        let mut job = self.with(py, |m| m.begin_solve())?;
        let raised: Mutex<Option<PyErr>> = Mutex::new(None);
        let cancel = &self.cancel;
        let outcome = py.detach(|| {
            let mut call = |p: &Progress| {
                let Some(f) = progress.as_ref() else {
                    return;
                };
                Python::attach(|py| match f.bind(py).call1((report(p),)) {
                    Ok(go_on) => {
                        if go_on.cast::<PyBool>().is_ok_and(|b| !b.is_true()) {
                            cancel.cancel();
                        }
                    }
                    Err(e) => {
                        // the first exception, raised once the solve has ended
                        raised.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert(e);
                        cancel.cancel();
                    }
                });
            };
            let callback: Option<&mut ProgressFn<'_>> = if progress.is_some() { Some(&mut call) } else { None };
            // a panic of the solve still ends it (finish_solve below), so that the model leaves
            // the calculating state and its progress ends
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(callback))).unwrap_or_else(|_| {
                Err(hydrobem_core::Error::internal("internal error of the engine in the solve (please report it)"))
            })
        });
        let raised = raised.into_inner().unwrap_or_else(|p| p.into_inner());
        let finished: Result<Results, hydrobem_core::Error> = {
            let mut g = self.lock(py);
            let m = g.as_mut().expect("a model is not closed while it calculates");
            m.finish_solve(job, outcome).cloned()
        };
        match finished {
            Ok(r) => Ok(PyResults::of(r)),
            Err(e) if e.kind() == ErrorKind::Cancelled && raised.is_some() => Err(raised.expect("an exception")),
            Err(e) => Err(e).py(),
        }
    }

    /// A copy of the results of the last solve (a RuntimeError when the model is not calculated).
    fn results(&self, py: Python<'_>) -> PyResult<PyResults> {
        self.with(py, |m| m.results().cloned()).map(PyResults::of)
    }

    /// Discards the results, the inputs kept.
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |m| m.reset())
    }

    /// Every input and the results back to a new model.
    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |m| m.clear())
    }

    // ---- the meshes the engine makes

    /// A mesh read from a GDF file (and its SPL file) by the rules of reference/inputs-meshes.
    #[pyo3(signature = (name, path, higher_order = false, zero_trailer = false, spl = None, discretization = None,
                        panel_size = None, spline_order = None, outer_quadrature = None, inner_quadrature = None))]
    #[allow(clippy::too_many_arguments)] // the keyword arguments of the Python method
    fn load_mesh(
        slf: &Bound<'_, Self>,
        name: &str,
        path: PathBuf,
        higher_order: bool,
        zero_trailer: bool,
        spl: Option<PathBuf>,
        discretization: Option<Vec<[i32; 6]>>,
        panel_size: Option<f64>,
        spline_order: Option<i32>,
        outer_quadrature: Option<i32>,
        inner_quadrature: Option<i32>,
    ) -> PyResult<PyObj> {
        let file = MeshFile {
            path,
            higher_order,
            zero_trailer,
            spl_path: spl,
            discretization,
            panel_size,
            spline_order,
            outer_quadrature,
            inner_quadrature,
        };
        PyModel::mesh(slf, slf.get().with(slf.py(), |m| m.load_mesh(name, &file)))
    }

    /// The panels of a free-surface file (FDF) as a panels mesh, with the model's symmetry.
    #[pyo3(signature = (name, path, symmetry = "none"))]
    fn load_fdf(slf: &Bound<'_, Self>, name: &str, path: PathBuf, symmetry: &str) -> PyResult<PyObj> {
        PyModel::mesh(slf, slf.get().with(slf.py(), |m| m.load_fdf(name, &path, symmetry)))
    }

    /// A control surface read from a CSF file, the automatic surface placed by body.
    #[pyo3(signature = (name, path, zero_trailer = false, body = None))]
    fn load_csf(
        slf: &Bound<'_, Self>,
        name: &str,
        path: PathBuf,
        zero_trailer: bool,
        body: Option<PyRef<'_, PyObj>>,
    ) -> PyResult<PyObj> {
        let body = PyModel::own(slf, body)?;
        PyModel::mesh(slf, slf.get().with(slf.py(), |m| m.load_csf(name, &path, zero_trailer, body)))
    }

    /// A mesh from the text of a geometry file (.hbg, reference/inputs-meshes §5).
    fn hbg_mesh(slf: &Bound<'_, Self>, name: &str, text: &str) -> PyResult<PyObj> {
        PyModel::mesh(slf, slf.get().with(slf.py(), |m| m.hbg_mesh(name, text)))
    }

    /// A named analytic shape (reference/inputs-meshes §2).
    #[pyo3(signature = (name, shape, dimensions, discs, parts, mapping, discretization, reference_length, symmetry, label))]
    #[allow(clippy::too_many_arguments)] // the keyword arguments of the Python method
    fn analytic_mesh(
        slf: &Bound<'_, Self>,
        name: &str,
        shape: String,
        dimensions: Vec<(String, f64)>,
        discs: Vec<[f64; 3]>,
        parts: Vec<String>,
        mapping: String,
        discretization: Vec<[i32; 6]>,
        reference_length: f64,
        symmetry: String,
        label: String,
    ) -> PyResult<PyObj> {
        let shape = AnalyticShape {
            shape,
            dimensions,
            discs,
            parts,
            mapping,
            discretization,
            reference_length,
            symmetry,
            label,
        };
        PyModel::mesh(slf, slf.get().with(slf.py(), |m| m.analytic_mesh(name, &shape)))
    }
}
