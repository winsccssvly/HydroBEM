//! NumPy arrays in and out: the engine's vectors as read-only arrays of a shape, the point
//! arrays of the queries as rows, and the answers of the panel, tessellation and quality queries as
//! dicts of arrays.

use hydrobem_core::{MeshElements, MeshQuality, Panels, Tessellation};
use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Marks an array read-only: the answers of the engine are not changed in place.
pub fn frozen<'py>(a: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    a.getattr("flags")?.setattr("writeable", false)?;
    Ok(a)
}

/// A read-only array of the elements v (C order) with the shape dims.
pub fn array<'py, T: Element + Clone>(py: Python<'py>, v: Vec<T>, dims: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let a = PyArray1::from_vec(py, v).reshape(dims)?;
    frozen(a.into_any())
}

// the rows of a query as one array [n, ...]
fn flat<'py, T: Element + Copy, const N: usize>(
    py: Python<'py>,
    v: &[[T; N]],
    dims: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    array(py, v.as_flattened().to_vec(), dims)
}

// the corners of a query as one array [n, 4, 3]
fn quads<'py>(py: Python<'py>, v: &[[[f64; 3]; 4]]) -> PyResult<Bound<'py, PyAny>> {
    array(py, v.as_flattened().as_flattened().to_vec(), &[v.len(), 4, 3])
}

/// The panels of a query as a dict of read-only arrays (the fields of `Panels`, the roles by value).
pub fn panels_dict<'py>(py: Python<'py>, p: &Panels) -> PyResult<Bound<'py, PyDict>> {
    let n = p.len();
    let x = PyDict::new(py);
    x.set_item("vertices", quads(py, &p.vertices)?)?;
    x.set_item("distinct_vertices", array(py, p.distinct_vertices.clone(), &[n])?)?;
    x.set_item("body", array(py, p.body.clone(), &[n])?)?;
    x.set_item("role", array(py, p.role.iter().map(|r| r.value()).collect(), &[n])?)?;
    x.set_item("tank", array(py, p.tank.clone(), &[n])?)?;
    x.set_item("owner", array(py, p.owner.clone(), &[n])?)?;
    x.set_item("element", array(py, p.element.clone(), &[n])?)?;
    x.set_item("part", array(py, p.part.clone(), &[n])?)?;
    x.set_item("reflection", flat(py, &p.reflection, &[n, 2])?)?;
    x.set_item("centroids", flat(py, &p.centroids, &[n, 3])?)?;
    x.set_item("normals", flat(py, &p.normals, &[n, 3])?)?;
    x.set_item("areas", array(py, p.areas.clone(), &[n])?)?;
    Ok(x)
}

/// The stored elements of a mesh as a dict of read-only arrays (the fields of `MeshElements`).
pub fn elements_dict<'py>(py: Python<'py>, r: &MeshElements) -> PyResult<Bound<'py, PyDict>> {
    let n = r.len();
    let x = PyDict::new(py);
    x.set_item("element", array(py, r.element.clone(), &[n])?)?;
    x.set_item("vertices", quads(py, &r.vertices)?)?;
    x.set_item("distinct_vertices", array(py, r.distinct_vertices.clone(), &[n])?)?;
    x.set_item("centroids", flat(py, &r.centroids, &[n, 3])?)?;
    x.set_item("normals", flat(py, &r.normals, &[n, 3])?)?;
    x.set_item("areas", array(py, r.areas.clone(), &[n])?)?;
    Ok(x)
}

/// A tessellation as a dict of read-only arrays (the fields of `Tessellation`, the roles by value,
/// -1 none).
pub fn tessellation_dict<'py>(py: Python<'py>, t: &Tessellation) -> PyResult<Bound<'py, PyDict>> {
    let n = t.len();
    let x = PyDict::new(py);
    x.set_item("vertices", quads(py, &t.vertices)?)?;
    x.set_item("normals", quads(py, &t.normals)?)?;
    x.set_item("body", array(py, t.body.clone(), &[n])?)?;
    x.set_item("role", array(py, t.role.iter().map(|r| r.map_or(-1, |r| r.value())).collect(), &[n])?)?;
    x.set_item("tank", array(py, t.tank.clone(), &[n])?)?;
    x.set_item("owner", array(py, t.owner.clone(), &[n])?)?;
    x.set_item("element", array(py, t.element.clone(), &[n])?)?;
    x.set_item("part", array(py, t.part.clone(), &[n])?)?;
    x.set_item("reflection", flat(py, &t.reflection, &[n, 2])?)?;
    Ok(x)
}

/// The mesh quality as a dict of read-only arrays (the fields of `MeshQuality`, the flags as bits).
pub fn quality_dict<'py>(py: Python<'py>, q: &MeshQuality) -> PyResult<Bound<'py, PyDict>> {
    let (n, b) = (q.body.len(), q.volumes.len());
    let x = PyDict::new(py);
    x.set_item("body", array(py, q.body.clone(), &[n])?)?;
    x.set_item("element", array(py, q.element.clone(), &[n])?)?;
    x.set_item("part", array(py, q.part.clone(), &[n])?)?;
    x.set_item("size", array(py, q.size.clone(), &[n])?)?;
    x.set_item("aspect", array(py, q.aspect.clone(), &[n])?)?;
    x.set_item("per_wavelength", array(py, q.per_wavelength.clone(), &[n])?)?;
    x.set_item("flags", array(py, q.flags.iter().map(|f| f.0 as i32).collect(), &[n])?)?;
    x.set_item("volumes", flat(py, &q.volumes, &[b, 3])?)?;
    x.set_item("min_per_wavelength", array(py, q.min_per_wavelength.clone(), &[b])?)?;
    x.set_item("edges", flat(py, &q.edges, &[b, 4])?)?;
    Ok(x)
}

/// The rows of an (n, N) float64 array.
pub fn rows<const N: usize>(a: &PyReadonlyArray2<'_, f64>, what: &str) -> PyResult<Vec<[f64; N]>> {
    let shape = a.shape();
    if shape[1] != N {
        return Err(PyValueError::new_err(format!("{what} must have the shape (n, {N})")));
    }
    let v = a.as_array();
    Ok(v.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect())
}
