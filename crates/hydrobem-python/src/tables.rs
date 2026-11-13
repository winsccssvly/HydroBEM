//! The engine's tables and the functions of the module that need no model: the schema of the model
//! objects, the result quantities, the build, the wave number and the names of the engine's codes.

use hydrobem_core::{DType, ElementType, ItemKind, PanelRole, QualityFlags, QuantityId, schema};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::errors::ToPy;

/// The NumPy name of an element type.
pub fn element_name(t: ElementType) -> &'static str {
    match t {
        ElementType::Float64 => "float64",
        ElementType::Float32 => "float32",
        ElementType::ComplexF32 => "complex64",
        ElementType::Int32 => "int32",
        ElementType::ComplexF64 => "complex128",
        ElementType::Int64 => "int64",
        ElementType::Text => "str",
    }
}

/// The schema: per object type its items (kind, dtype, shape, unit, default, choices,
/// target, mesh kinds, the switch items that choose it and their values that take it, description)
/// and its lists (types, named, optional_names, single, always, description).
#[pyfunction(name = "schema")]
fn schema_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for t in schema() {
        let items = PyDict::new(py);
        for d in &t.items {
            let x = PyDict::new(py);
            x.set_item("kind", d.kind.name())?;
            x.set_item("dtype", if d.kind == ItemKind::Array { d.dtype } else { DType::Real }.name())?;
            x.set_item("shape", d.shape)?;
            x.set_item("unit", d.unit)?;
            x.set_item("default", d.fallback)?;
            x.set_item("choices", PyTuple::new(py, &d.choices)?)?;
            x.set_item("target", d.target)?;
            x.set_item("kinds", PyTuple::new(py, d.kinds.split_whitespace().collect::<Vec<_>>())?)?;
            x.set_item("switch", PyTuple::new(py, d.switch.split_whitespace().collect::<Vec<_>>())?)?;
            x.set_item("switch_values", PyTuple::new(py, d.switch_values.split_whitespace().collect::<Vec<_>>())?)?;
            x.set_item("description", d.description)?;
            items.set_item(d.name, x)?;
        }
        let lists = PyDict::new(py);
        for l in &t.lists {
            let x = PyDict::new(py);
            x.set_item("types", PyTuple::new(py, &l.types)?)?;
            x.set_item("named", l.named)?;
            x.set_item("optional_names", l.optional_names)?;
            x.set_item("single", l.single)?;
            x.set_item("always", l.always)?;
            x.set_item("description", l.description)?;
            lists.set_item(l.name, x)?;
        }
        let entry = PyDict::new(py);
        entry.set_item("items", items)?;
        entry.set_item("lists", lists)?;
        out.set_item(t.name, entry)?;
    }
    Ok(out)
}

/// The result quantities in the order of their codes: name, NumPy dtype, axes, unit, description.
#[pyfunction]
fn quantities(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    QuantityId::ALL
        .iter()
        .map(|q| {
            let x = PyDict::new(py);
            x.set_item("name", q.name())?;
            x.set_item("dtype", element_name(q.element_type()))?;
            x.set_item("dims", PyTuple::new(py, q.dims().split(',').collect::<Vec<_>>())?)?;
            x.set_item("unit", q.unit())?;
            x.set_item("description", q.description())?;
            Ok(x)
        })
        .collect()
}

/// The views of the result tree in the order of the quantity table: path, quantity (its name),
/// code, owner (the rule's name) and width (the columns of a body of the rule "columns", else 0).
#[pyfunction]
fn result_views(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    hydrobem_core::result_views()
        .iter()
        .map(|v| {
            let x = PyDict::new(py);
            x.set_item("path", v.path)?;
            x.set_item("quantity", v.quantity.name())?;
            x.set_item("code", QuantityId::ALL.iter().position(|q| *q == v.quantity).unwrap_or(usize::MAX))?;
            x.set_item("owner", v.owner.name())?;
            x.set_item("width", v.owner.width())?;
            Ok(x)
        })
        .collect()
}

/// The groups of the result tree, parents first: path, description.
#[pyfunction]
fn result_groups(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    hydrobem_core::result_groups()
        .iter()
        .map(|g| {
            let x = PyDict::new(py);
            x.set_item("path", g.path)?;
            x.set_item("description", g.description)?;
            Ok(x)
        })
        .collect()
}

/// The build of the engine.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let b = hydrobem_core::build_info();
    let x = PyDict::new(py);
    x.set_item("version", &b.version)?;
    x.set_item("commit", &b.commit)?;
    x.set_item("compiler", &b.compiler)?;
    x.set_item("linear_algebra", &b.linear_algebra)?;
    x.set_item("build_type", &b.build_type)?;
    x.set_item("api_version", b.api_version)?;
    x.set_item("avx2", b.avx2)?;
    Ok(x)
}

/// The wave number of omega^2 = g k tanh(k h).
#[pyfunction]
fn wavenumber(omega: f64, depth: f64, gravity: f64) -> PyResult<f64> {
    hydrobem_core::wavenumber(omega, depth, gravity).py()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_function(wrap_pyfunction!(schema_info, m)?)?;
    m.add_function(wrap_pyfunction!(quantities, m)?)?;
    m.add_function(wrap_pyfunction!(result_views, m)?)?;
    m.add_function(wrap_pyfunction!(result_groups, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(wavenumber, m)?)?;
    // the roles of the panels by their values (the "role" array of a panels answer)
    let roles: Vec<&str> = (0..).map_while(PanelRole::from_value).map(PanelRole::name).collect();
    m.add("PANEL_ROLES", PyTuple::new(py, roles)?)?;
    // the flags of the mesh quality of an element by name
    let flags = PyDict::new(py);
    for (name, f) in [
        ("orientation", QualityFlags::ORIENTATION),
        ("free_edge", QualityFlags::FREE_EDGE),
        ("non_manifold", QualityFlags::NON_MANIFOLD),
        ("coarse", QualityFlags::COARSE),
    ] {
        flags.set_item(name, f.0)?;
    }
    m.add("QUALITY_FLAGS", flags)?;
    Ok(())
}
