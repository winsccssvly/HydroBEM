//! `Object`: an object of a model, its model and its `ObjectId`. Its items are read as
//! Python values (bool, int, float, str, a tuple of str, a read-only array) and set from them with
//! the checks of their kind (a TypeError names the item); the engine checks the value. An
//! object taken out of its model is refused by the engine; one of a closed model here.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use hydrobem_core::{DType, ItemDef, ItemKind, MeshGeometry, ObjectId, Value};
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString, PyTuple};

use crate::convert::{array, elements_dict, panels_dict, quality_dict, rows, tessellation_dict};
use crate::model::{PyModel, hydrostatics_dict};

const CLOSED: &str = "the model of this object is closed";

#[pyclass(frozen, name = "Object", module = "hydrobem._core")]
pub struct PyObj {
    pub(crate) model: Py<PyModel>,
    pub(crate) id: ObjectId,
}

/// A name of a new or renamed object: a string.
fn name_of(name: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    match name {
        None => Ok(String::new()),
        Some(n) if n.is_none() => Ok(String::new()),
        Some(n) => match n.cast::<PyString>() {
            Ok(s) => Ok(s.to_str()?.to_string()),
            Err(_) => Err(PyTypeError::new_err(format!("a name must be a string, got {}", n.get_type().name()?))),
        },
    }
}

fn is_texty(k: ItemKind) -> bool {
    matches!(k, ItemKind::Text | ItemKind::Choice | ItemKind::Reference)
}

/// The value of an item for a Python value, by the item's kind: the TypeErrors of a value of
/// another kind, each naming the item (`at` its object's path).
fn value_of(py: Python<'_>, d: &ItemDef, at: &str, item: &str, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let np = py.import("numpy")?;
    let is_bool = value.is_instance_of::<PyBool>() || value.is_instance(&np.getattr("bool_")?)?;
    let is_int = !value.is_instance_of::<PyBool>()
        && (value.is_instance_of::<PyInt>() || value.is_instance(&np.getattr("integer")?)?);
    let is_real = is_int || value.is_instance_of::<PyFloat>() || value.is_instance(&np.getattr("floating")?)?;
    let fail = |what: &str| Err(PyTypeError::new_err(format!("{at}{item} must be {what}")));
    // a number item takes one of its names instead
    let names = if matches!(d.kind, ItemKind::Int | ItemKind::Real) { d.choices.as_slice() } else { &[] };
    if !names.is_empty() && value.is_instance_of::<PyString>() {
        return Ok(Value::Text(value.extract()?));
    }
    let alt = if names.is_empty() { String::new() } else { format!(" or one of {}", names.join(", ")) };
    match d.kind {
        ItemKind::Bool => {
            if !is_bool {
                return fail("a bool");
            }
            Ok(Value::Bool(value.is_truthy()?))
        }
        ItemKind::Int => {
            if !is_int {
                return fail(&format!("an integer{alt}"));
            }
            Ok(Value::Int(value.extract()?))
        }
        ItemKind::Real => {
            if !is_real {
                return fail(&format!("a real number{alt}"));
            }
            Ok(Value::Real(value.extract()?))
        }
        k if is_texty(k) || (k == ItemKind::Origin && value.is_instance_of::<PyString>()) => {
            if !value.is_instance_of::<PyString>() {
                return fail("a string");
            }
            Ok(Value::Text(value.extract()?))
        }
        _ => array_value(py, d, at, item, value),
    }
}

/// The value of an array item (an array, terms or an origin given by its coordinates).
fn array_value(py: Python<'_>, d: &ItemDef, at: &str, item: &str, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let fail = |what: &str| Err(PyTypeError::new_err(format!("{at}{item} must be {what}")));
    let dtype = if d.kind == ItemKind::Array { d.dtype } else { DType::Real };
    if dtype == DType::Text {
        if value.is_instance_of::<PyString>() {
            return fail("a sequence of strings");
        }
        // once: a generator is read here
        let Ok(texts) = py.import("builtins")?.getattr("list")?.call1((value,)) else {
            return fail("a sequence of strings");
        };
        let mut out = Vec::new();
        for t in texts.try_iter()? {
            let t = t?;
            if !t.is_instance_of::<PyString>() {
                return fail("a sequence of strings");
            }
            out.push(t.extract()?);
        }
        return Ok(Value::Texts(out));
    }
    let np = py.import("numpy")?;
    let a = np.call_method1("asarray", (value,))?;
    // before the conversions below, which make a scalar one element
    if a.getattr("ndim")?.extract::<usize>()? == 0 {
        return fail("an array");
    }
    if np.call_method1("iscomplexobj", (&a,))?.is_truthy()? {
        return fail("real");
    }
    let kind: String = a.getattr("dtype")?.getattr("kind")?.extract()?;
    let shape: Vec<usize> = a.getattr("shape")?.extract()?;
    if "iu".contains(kind.as_str()) && dtype == DType::Int {
        let c = np.call_method1("ascontiguousarray", (&a, np.getattr("int64")?))?;
        let v = c.cast::<PyArrayDyn<i64>>()?.readonly();
        return Ok(Value::Ints { shape, data: v.as_array().iter().copied().collect() });
    }
    if kind.len() == 1 && "iuf".contains(kind.as_str()) {
        if dtype == DType::Int {
            return fail("an array of integers");
        }
        let c = np.call_method1("ascontiguousarray", (&a, np.getattr("float64")?))?;
        let v = c.cast::<PyArrayDyn<f64>>()?.readonly();
        return Ok(Value::Reals { shape, data: v.as_array().iter().copied().collect() });
    }
    fail("an array of numbers")
}

/// A Python value of an item's value.
fn python_value<'py>(py: Python<'py>, v: Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Bool(b) => PyBool::new(py, b).to_owned().into_any(),
        Value::Int(x) => x.into_pyobject(py)?.into_any(),
        Value::Real(x) => PyFloat::new(py, x).into_any(),
        Value::Text(t) => PyString::new(py, &t).into_any(),
        Value::Texts(t) => PyTuple::new(py, t)?.into_any(),
        Value::Reals { shape, data } => array(py, data, &shape)?,
        Value::Ints { shape, data } => array(py, data, &shape)?,
    })
}

impl PyObj {
    pub fn new(model: Py<PyModel>, id: ObjectId) -> PyObj {
        PyObj { model, id }
    }

    fn with<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut hydrobem_core::Model) -> hydrobem_core::Result<R>,
    ) -> PyResult<R> {
        self.model.get().with_or(py, CLOSED, f)
    }

    fn child(&self, py: Python<'_>, id: ObjectId) -> PyObj {
        PyObj { model: self.model.clone_ref(py), id }
    }
}

#[pymethods]
impl PyObj {
    /// The model of the object.
    #[getter]
    fn model(&self, py: Python<'_>) -> Py<PyModel> {
        self.model.clone_ref(py)
    }

    #[getter(r#type)]
    fn type_name(&self, py: Python<'_>) -> PyResult<&'static str> {
        self.with(py, |m| m.object_info(self.id).map(|i| i.type_name))
    }

    #[getter]
    fn name(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |m| m.object_info(self.id).map(|i| i.name))
    }

    #[getter]
    fn path(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |m| m.path(self.id))
    }

    /// The objects of a list of the object.
    fn children(&self, py: Python<'_>, list: &str) -> PyResult<Vec<PyObj>> {
        let ids = self.with(py, |m| m.children(self.id, list).map(|v| v.to_vec()))?;
        Ok(ids.into_iter().map(|id| self.child(py, id)).collect())
    }

    /// A new object of a list: of a type (None: the list's first), with a name in a named list, at a
    /// zero-based place (-1: the end).
    #[pyo3(signature = (list, name = None, r#type = None, index = -1))]
    fn add(
        &self,
        py: Python<'_>,
        list: &str,
        name: Option<&Bound<'_, PyAny>>,
        r#type: Option<&str>,
        index: i64,
    ) -> PyResult<PyObj> {
        let name = name_of(name)?;
        let at = if index < 0 { None } else { Some(usize::try_from(index).unwrap_or(usize::MAX)) };
        let id = self.with(py, |m| m.add(self.id, list, r#type, &name, at))?;
        Ok(self.child(py, id))
    }

    /// Removes the object and its children.
    fn remove(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |m| m.remove(self.id))
    }

    /// Removes every object of a list, all or nothing (refused while another object refers to
    /// one of them).
    fn clear(&self, py: Python<'_>, list: &str) -> PyResult<()> {
        self.with(py, |m| m.clear_list(self.id, list))
    }

    fn rename(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = name_of(Some(name))?;
        self.with(py, |m| m.rename(self.id, &name))
    }

    /// Moves the object to a zero-based place of its list.
    #[pyo3(name = "move")]
    fn move_to(&self, py: Python<'_>, index: i64) -> PyResult<()> {
        let at = usize::try_from(index).unwrap_or(usize::MAX); // out of range for the engine
        self.with(py, |m| m.move_to(self.id, at))
    }

    /// The kind and state of an item: kind, is_set, has_value, is_array, dtype, shape, is_choice.
    fn describe<'py>(&self, py: Python<'py>, item: &str) -> PyResult<Bound<'py, PyDict>> {
        let i = self.with(py, |m| m.item_info(self.id, item))?;
        let x = PyDict::new(py);
        x.set_item("kind", i.kind.name())?;
        x.set_item("is_set", i.is_set)?;
        x.set_item("has_value", i.has_value)?;
        x.set_item("is_array", i.is_array)?;
        x.set_item("dtype", i.dtype.name())?;
        x.set_item("shape", PyTuple::new(py, &i.shape)?)?;
        x.set_item("is_choice", i.is_choice)?;
        Ok(x)
    }

    /// The value of an item (its default when not set): bool, int, float, str, a tuple of str, a
    /// read-only array; None when it has no value.
    fn get<'py>(&self, py: Python<'py>, item: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.with(py, |m| m.get(self.id, item))? {
            None => Ok(None),
            Some(v) => python_value(py, v).map(Some),
        }
    }

    /// Sets an item (checked by the engine); None resets it.
    fn set(&self, py: Python<'_>, item: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            return self.with(py, |m| m.reset_item(self.id, item));
        }
        // the definition of the item (static: the type of an object never changes) and the place
        // for the messages, then the value converted with the model unlocked (the conversion runs
        // Python code, which may call the model again), then the set, which refuses an object
        // removed meanwhile
        let (def, at) = self.with(py, |m| {
            m.object_info(self.id)?; // an object of the model
            let o = m.document().get_obj(self.id).expect("an object of the model");
            Ok((o.item_def(item)?, o.prefix()))
        })?;
        let v = value_of(py, def, &at, item, value)?;
        self.with(py, |m| m.set(self.id, item, v))
    }

    /// Returns an item to its default (or unsets it).
    fn reset(&self, py: Python<'_>, item: &str) -> PyResult<()> {
        self.with(py, |m| m.reset_item(self.id, item))
    }

    // ---- the functions of the engine on a body (after the model rules)

    /// A new mesh of the model: the surface of one role of this body as the engine uses it (after
    /// its trim).
    fn trimmed_mesh(&self, py: Python<'_>, name: &str, role: &str) -> PyResult<PyObj> {
        let id = self.with(py, |m| m.body_trimmed_mesh(self.id, name, role))?;
        Ok(self.child(py, id))
    }

    /// The hydrostatics of this body after its trim and placement (theory §3.2).
    fn body_hydrostatics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let h = self.with(py, |m| m.body_hydrostatics(self.id))?;
        hydrostatics_dict(py, &h)
    }

    /// The elements of this body as the engine uses them, of the roles named (none: the default
    /// roles): a dict of arrays.
    fn body_panels<'py>(
        &self,
        py: Python<'py>,
        include_reflections: bool,
        roles: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let roles: Vec<&str> = roles.iter().map(String::as_str).collect();
        panels_dict(py, &self.with(py, |m| m.body_panels(self.id, &roles, include_reflections))?)
    }

    /// The quality of the mesh of this body: a dict of arrays.
    fn body_mesh_quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        quality_dict(py, &self.with(py, |m| m.body_mesh_quality(self.id))?)
    }

    /// The tessellation of the elements of this body as the engine uses them, placed: a dict of
    /// arrays.
    fn body_tessellation<'py>(
        &self,
        py: Python<'py>,
        roles: Vec<String>,
        divisions: [i32; 2],
        include_reflections: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let roles: Vec<&str> = roles.iter().map(String::as_str).collect();
        tessellation_dict(py, &self.with(py, |m| m.body_tessellation(self.id, &roles, divisions, include_reflections))?)
    }

    // ---- the functions of the engine on a mesh (its geometry alone, mesh coordinates)

    /// The text of the geometry file (.hbg) of the mesh.
    fn hbg(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |m| m.mesh_hbg(self.id))
    }

    /// Writes the mesh as GDF (and SPL).
    #[pyo3(signature = (gdf, spl = None, overwrite = false))]
    fn write(&self, py: Python<'_>, gdf: PathBuf, spl: Option<PathBuf>, overwrite: bool) -> PyResult<()> {
        self.with(py, |m| m.mesh_write(self.id, &gdf, spl.as_deref(), overwrite))
    }

    /// The geometry of the mesh: volumes, volume, center_of_volume, area, bounds, the waterplane.
    fn mesh_geometry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g: MeshGeometry = self.with(py, |m| m.mesh_geometry(self.id))?;
        let x = PyDict::new(py);
        x.set_item("volumes", g.volumes.to_vec())?;
        x.set_item("volume", g.volume)?;
        x.set_item("center_of_volume", g.center_of_volume.to_vec())?;
        x.set_item("area", g.area)?;
        x.set_item("bounds", array(py, g.bounds.concat(), &[2, 3])?)?;
        x.set_item("waterplane_area", g.waterplane_area)?;
        x.set_item("waterplane_centroid", g.waterplane_centroid.to_vec())?;
        x.set_item("waterplane_moments", g.waterplane_moments.to_vec())?;
        Ok(x)
    }

    /// The stored elements of the mesh: a dict of arrays.
    fn mesh_elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        elements_dict(py, &self.with(py, |m| m.mesh_elements(self.id))?)
    }

    /// The tessellation of the mesh alone: a dict of arrays.
    fn mesh_tessellation<'py>(
        &self,
        py: Python<'py>,
        divisions: [i32; 2],
        include_reflections: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        tessellation_dict(py, &self.with(py, |m| m.mesh_tessellation(self.id, divisions, include_reflections))?)
    }

    /// The geometric quality of the mesh alone: a dict of arrays.
    fn mesh_geometry_quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        quality_dict(py, &self.with(py, |m| m.mesh_geometry_quality(self.id))?)
    }

    /// Points and unit normals of a patch at parameters uv [n, 2] (mesh coordinates).
    fn patch_points<'py>(
        &self,
        py: Python<'py>,
        element: i32,
        uv: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let uv = rows::<2>(&uv, "uv")?;
        let p = self.with(py, |m| m.mesh_patch_points(self.id, element, &uv))?;
        let n = p.points.len();
        Ok((array(py, p.points.concat(), &[n, 3])?, array(py, p.normals.concat(), &[n, 3])?))
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        match other.cast::<PyObj>() {
            Ok(o) => {
                let o = o.get();
                o.id == self.id && o.model.bind(py).is(self.model.bind(py))
            }
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.model.as_ptr().hash(&mut h);
        self.id.hash(&mut h);
        h.finish()
    }
}
