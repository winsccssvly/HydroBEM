//! `Results`: the results of a solve, of a results file (each quantity read when first asked
//! for), or made from arrays. Its quantities are NumPy arrays over the
//! engine's elements (read-only; the array keeps the results alive).

use std::path::PathBuf;

use hydrobem_core::{Complex64, QuantityData, QuantityId, Results, Rows, WamitOptions};
use numpy::ndarray::{ArrayView, IxDyn};
use numpy::{Element, PyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use crate::convert::{array, frozen};
use crate::errors::{ToPy, diagnostics};
use crate::model::PyModel;

#[pyclass(frozen, name = "Results", module = "hydrobem._core")]
pub struct PyResults {
    pub(crate) r: Results,
}

impl PyResults {
    pub fn of(r: Results) -> PyResults {
        PyResults { r }
    }
}

/// The quantity of a code (the row of the quantity table).
pub fn quantity_id(code: usize) -> PyResult<QuantityId> {
    QuantityId::ALL.get(code).copied().ok_or_else(|| PyValueError::new_err(format!("no result quantity {code}")))
}

/// A read-only array over the elements v of the results `owner`, of the shape dims.
fn borrowed<'py, T: Element>(owner: &Bound<'py, PyResults>, v: &[T], dims: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let view = ArrayView::from_shape(IxDyn(dims), v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    // SAFETY: the elements belong to the results of owner, which the array holds as its base; a
    // frozen pyclass is never changed, so the elements stay where they are while it lives
    let a = unsafe { PyArray::borrow_from_array(&view, owner.clone().into_any()) };
    frozen(a.into_any())
}

// a selection as Python takes it: the quantity's code and the rows of each axis selected
type Selected<'py> = (usize, Vec<(usize, Bound<'py, PyAny>)>);

// zero-based indices as a NumPy intp array
fn intp(py: Python<'_>, v: Vec<usize>) -> Bound<'_, PyAny> {
    numpy::PyArray1::from_vec(py, v.into_iter().map(|x| x as isize).collect()).into_any()
}

#[pymethods]
impl PyResults {
    /// The quantity of a code as a read-only array; None when the results do not have it.
    fn quantity<'py>(slf: &Bound<'py, Self>, code: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
        let r = &slf.get().r;
        let q = quantity_id(code)?;
        let Some(shape) = r.shape(q) else {
            return Ok(None);
        };
        let dims = shape.dims;
        let a = match r.typed(q).py()? {
            QuantityData::Float64(v) => borrowed(slf, v, &dims)?,
            QuantityData::Float32(v) => borrowed(slf, v, &dims)?,
            QuantityData::ComplexF32(v) => borrowed(slf, v, &dims)?,
            QuantityData::Int32(v) => borrowed(slf, v, &dims)?,
            QuantityData::ComplexF64(v) => borrowed(slf, v, &dims)?,
            QuantityData::Int64(v) => borrowed(slf, v, &dims)?,
        };
        Ok(Some(a))
    }

    /// What the view of a path selects for an owner (`Results::selection`): None, or the code of
    /// the quantity and per axis selected (axis, rows): a slice for contiguous rows, else an intp
    /// array of the zero-based rows.
    #[pyo3(signature = (path, body = None, owner = None))]
    fn selection<'py>(
        &self,
        py: Python<'py>,
        path: &str,
        body: Option<usize>,
        owner: Option<usize>,
    ) -> PyResult<Option<Selected<'py>>> {
        let Some(s) = self.r.selection(path, body, owner).py()? else {
            return Ok(None);
        };
        let code = QuantityId::ALL.iter().position(|q| *q == s.quantity).unwrap_or(usize::MAX);
        let mut axes = Vec::with_capacity(s.axes.len());
        for (axis, rows) in s.axes {
            let r = match rows {
                Rows::Range(r) => PySlice::new(py, r.start as isize, r.end as isize, 1).into_any(),
                Rows::Indices(v) => intp(py, v),
            };
            axes.push((axis, r));
        }
        Ok(Some((code, axes)))
    }

    /// The zero-based indices of a body's coordinates (D axis) and radiation columns (K axis).
    fn body_columns<'py>(&self, py: Python<'py>, body: usize) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let (d, k) = self.r.body_columns(body).py()?;
        Ok((intp(py, d), intp(py, k)))
    }

    /// The codes of the quantities the results have, in the order of the table.
    fn available(&self) -> Vec<usize> {
        let codes = self.r.available();
        QuantityId::ALL.iter().enumerate().filter(|(_, q)| codes.contains(q)).map(|(k, _)| k).collect()
    }

    /// The solver route ("given" for results made from arrays).
    #[getter]
    fn route(&self) -> &str {
        self.r.route()
    }

    /// The diagnostics: the warnings of the model, then of the solve.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        diagnostics(py, self.r.diagnostics())
    }

    /// The meta data of the solve; None for results not from a solve.
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let m = self.r.meta();
        if !m.present {
            return Ok(None);
        }
        let x = PyDict::new(py);
        x.set_item("hydrobem_version", &m.hydrobem_version)?;
        x.set_item("api_version", m.api_version)?;
        x.set_item("commit", &m.commit)?;
        x.set_item("compiler", &m.compiler)?;
        x.set_item("linear_algebra", &m.linear_algebra)?;
        x.set_item("build_type", &m.build_type)?;
        x.set_item("time", &m.time)?;
        x.set_item("route", &m.route)?;
        x.set_item("frequency_threads", m.frequency_threads)?;
        x.set_item("inner_threads", m.inner_threads)?;
        x.set_item("engine_seconds", PyTuple::new(py, &m.engine_seconds)?)?;
        x.set_item("environment_hooks", PyTuple::new(py, &m.environment_hooks)?)?;
        Ok(Some(x))
    }

    /// A new model of the inputs of these results (a RuntimeError for results without a model).
    fn model(&self) -> PyResult<PyModel> {
        match self.r.document() {
            Some(doc) => Ok(PyModel::of(hydrobem_core::Model::with_document(doc.clone()))),
            None => Err(hydrobem_core::Error::state("these results hold no model")).py(),
        }
    }

    /// A new model of the model as it was solved: the enabled objects of the inputs, whose bodies,
    /// lids and field points the quantities number (a RuntimeError for results without a model).
    fn solved_model(&self) -> PyResult<PyModel> {
        match self.r.solved_document() {
            Some(doc) => Ok(PyModel::of(hydrobem_core::Model::with_document(doc.as_ref().clone()))),
            None => Err(hydrobem_core::Error::state("these results hold no model")).py(),
        }
    }

    /// A quantity normalized as in the numeric files (reference/file-formats §1): float64 or complex128.
    fn normalized<'py>(&self, py: Python<'py>, code: usize) -> PyResult<Bound<'py, PyAny>> {
        let q = quantity_id(code)?;
        let (dims, v) = self.r.normalized(q).py()?;
        let dims: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        if q.element_type().is_complex() {
            array::<Complex64>(py, v, &dims)
        } else {
            array::<f64>(py, v.iter().map(|z| z.re).collect(), &dims)
        }
    }

    /// Writes the results file (.hbr).
    #[pyo3(signature = (path, overwrite = false))]
    fn save(&self, py: Python<'_>, path: PathBuf, overwrite: bool) -> PyResult<()> {
        py.detach(|| self.r.save(&path, overwrite)).py()
    }

    /// Writes the WAMIT-format numeric files and the .out summary (reference/file-formats §2-§4).
    #[pyo3(signature = (stem, overwrite = false, omega = false, stamp = None))]
    fn write_wamit(
        &self,
        py: Python<'_>,
        stem: PathBuf,
        overwrite: bool,
        omega: bool,
        stamp: Option<String>,
    ) -> PyResult<()> {
        let options = WamitOptions { overwrite, omega, stamp };
        py.detach(|| self.r.write_wamit(&stem, &options)).py().map(|_| ())
    }
}
