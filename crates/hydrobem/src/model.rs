//! The model of the public API: `Model` owns the model document, the problem built from it (a
//! cache, rebuilt after a change), the results of its last solve and its state.
//!
//! - Revisions: every change of the document moves its revision on; the problem is rebuilt when
//!   its revision is not the document's, and the results of a solve are kept until an input
//!   changes.
//! - A solve runs as a `SolveJob` (`Model::begin_solve`, `SolveJob::run`, `Model::finish_solve`;
//!   `Model::solve` does all three): the job takes the built problem out of the model, so that the
//!   model can still be read while it runs (on another thread, or from the progress callback); the
//!   model is calculating until the job ends, and its inputs cannot change in that time.
//! - The cancel request of a solve is a `CancelToken` of the model (`Model::cancel_token`), usable
//!   from any thread; a request sent before a solve applies to it, and it is cleared when the solve
//!   ends.
//! - The objects of the document are addressed by `ObjectId` (document.rs); their items are read and
//!   set as `Value`s.
//!
//! This file holds the model itself: its construction, files, state, the problem built from it and
//! its objects and items. The other methods of `Model` are in model_queries.rs (the queries),
//! model_solve.rs (the solve, `SolveJob`), model_meshes.rs (the meshes as objects) and model_trim.rs
//! (the trimmed meshes); the small public types in model_types.rs, re-exported here.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::codes::ModelState;
use crate::document::{DType, Document, ItemKind, ItemValue, MAX_ARRAY_ELEMENTS, Obj, ObjectId};
use crate::document_build::build_scoped;
use crate::document_enabled::{Enabled, Numbering, Scope, scoped_document};
use crate::error::{Diagnostic, Error, Result};
use crate::problem::{CancelFlag, Environment, Options, Problem};
use crate::results::Results;
use crate::results_file::{open_results, save_model, save_results};

pub use crate::model_solve::SolveJob;
pub use crate::model_types::{
    AnalyticShape, BodyCounts, Counts, FreeSurfacePoints, ItemInfo, MeshFile, MotionEquation, ObjectInfo, PatchPoints,
    PointClass, Progress, ProgressAt, ProgressFn, Value, wavenumber,
};

/// A path as the engine's text (UTF-8).
pub(crate) fn path_text(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| Error::argument(format!("{}: the path is not valid UTF-8", p.display())))
}

pub use crate::progress::{
    CancelToken, FrequencyProgress, FrequencyState, Limit, LimitProgress, Phase, ProgressMonitor, SecondOrderProgress,
    SecondOrderStep, SolveProgress, Task, WorkerProgress, WorkerState,
};

// ------------------------------------------------------------------ helpers of the model

/// The inputs of a problem emptied for a new build.
pub(crate) fn reset_inputs(m: &mut Problem) {
    m.meshes.clear();
    m.origins.clear();
    m.bodies.clear();
    m.lids.clear();
    m.fixed_dampers.clear();
    m.env = Environment::default();
    m.opt = Options::default();
    m.omegas.clear();
    m.headings.clear();
    m.has_field_points = false;
    m.field_points.clear();
    m.field_tanks.clear();
    m.tank_domain = false;
    m.interior_domain = false;
    m.warnings.clear(); // the warnings of earlier inputs
    m.invalidate();
}

/// The kind of an item in the messages of a value of another kind.
pub(crate) fn kind_text(k: ItemKind) -> &'static str {
    match k {
        ItemKind::Bool => "a bool",
        ItemKind::Int => "an integer",
        ItemKind::Real => "a real",
        ItemKind::Array => "an array",
        ItemKind::Terms => "a terms array",
        ItemKind::Origin => "an origin",
        _ => "a text",
    }
}

fn is_texty(k: ItemKind) -> bool {
    k == ItemKind::Text || k == ItemKind::Choice || k == ItemKind::Reference
}

// ------------------------------------------------------------------ the model

/// A model: its document, the problem built from it, the results of its last solve and its
/// state.
pub struct Model {
    pub(crate) doc: Document,
    pub(crate) problem: Problem,
    pub(crate) built_revision: u64, // the revision of doc that problem was built from
    // the copy of the enabled objects that problem was built from, and its numbering
    pub(crate) enabled: Option<Enabled>,
    // the warnings of the disabled objects of the last check (Model::check)
    pub(crate) disabled_warnings: Vec<Diagnostic>,
    // the problem of the last query of a body or of the free surface (Model::scoped_query), kept
    // until an input changes or a query of another scope
    pub(crate) scoped: Option<ScopedProblem>,
    // the results of the last solve and the revision of doc they belong to
    pub(crate) results: Option<Arc<Results>>,
    pub(crate) results_revision: u64,
    pub(crate) calculating: Arc<AtomicBool>,
    pub(crate) cancel: CancelToken,
    pub(crate) monitor: ProgressMonitor,
}

/// The problem of the queries of a scope (`Scope::Body`, `Scope::FreeSurface`: built from
/// `document_enabled::scoped_document` by `document_build::build_scoped`, its rules checked), the
/// revision of the document it was built from and the numbering of its copy in the model.
pub(crate) struct ScopedProblem {
    pub(crate) revision: u64,
    pub(crate) scope: Scope,
    pub(crate) numbering: Numbering,
    pub(crate) problem: Box<Problem>,
    /// the index in the problem of the body of a body's scope (0, or after the bodies over whose
    /// waterline loops its generated plates lie, kept in the model's order); 0 in another scope
    pub(crate) index: i32,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("revision", &self.doc.revision()).field("state", &self.state()).finish()
    }
}

impl Default for Model {
    fn default() -> Self {
        Model::new()
    }
}

impl Model {
    /// An empty model.
    pub fn new() -> Model {
        Model::with_document(Document::new())
    }
    /// A model of a document (its inputs; no results).
    pub fn with_document(doc: Document) -> Model {
        let cancel = CancelToken::new();
        let problem = Problem { cancel: CancelFlag::shared(cancel.clone()), ..Problem::default() };
        Model {
            doc,
            problem,
            built_revision: u64::MAX,
            enabled: None,
            disabled_warnings: Vec::new(),
            scoped: None,
            results: None,
            results_revision: u64::MAX,
            calculating: Arc::new(AtomicBool::new(false)),
            cancel,
            monitor: ProgressMonitor::new(),
        }
    }
    /// An independent copy of the inputs of the model (no results).
    pub fn clone_inputs(&self) -> Model {
        Model::with_document(self.doc.clone())
    }
    /// Reads a model file (.hbm), or the inputs of a results file (.hbr), into a new model: every
    /// item set again, so that it passes the checks it passed when given.
    pub fn load(path: &Path) -> Result<Model> {
        let p = path_text(path)?;
        // a model file (.hbm) or the inputs of a results file (.hbr)
        let r = open_results(p)?;
        let Some(doc) = r.model.as_ref() else {
            return Err(Error::value(format!("{p}: the file holds no model")));
        };
        Ok(Model::with_document(Document::clone(doc)))
    }
    /// Writes the model as a model file (.hbm): every object and item, the meshes
    /// included, and the build and time of the save; the folder made. ErrorKind::Exists when the file
    /// exists and overwrite is false.
    pub fn save(&self, path: &Path, overwrite: bool) -> Result<()> {
        save_model(&self.doc, path_text(path)?, overwrite)
    }
    /// A calculated model from a results file: its inputs and its results (each quantity read when
    /// first used).
    pub fn load_results(path: &Path) -> Result<Model> {
        let p = path_text(path)?;
        let r = open_results(p)?;
        if r.model_file {
            return Err(Error::value(format!("{p}: a model file (.hbm) holds no results (open it as a model)")));
        }
        let Some(doc) = r.model.clone() else {
            return Err(Error::value(format!("{p}: the results file holds no model")));
        };
        let mut m = Model::with_document(Document::clone(&doc));
        m.results_revision = m.doc.revision();
        m.results = Some(Arc::new(r));
        Ok(m)
    }
    /// Writes the model with its results as a results file (.hbr).
    pub fn save_results(&self, path: &Path, overwrite: bool) -> Result<()> {
        let Some(r) = self.results.as_ref().filter(|_| self.calculated()) else {
            return Err(Error::state("the model has no results to save (save writes its inputs)"));
        };
        save_results(r, path_text(path)?, overwrite)
    }

    // ---- the state

    /// The state: reset, calculating or calculated.
    pub fn state(&self) -> ModelState {
        if self.calculating.load(Ordering::SeqCst) {
            ModelState::Calculating
        } else if self.calculated() {
            ModelState::Calculated
        } else {
            ModelState::Reset
        }
    }
    // the model holds results of its present inputs
    fn calculated(&self) -> bool {
        self.results.is_some() && self.results_revision == self.doc.revision()
    }
    /// The cancel request of the solves of this model.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }
    /// The progress of the model's solves, to be read from any thread while one runs
    /// (`ProgressMonitor::snapshot`; the model need not be locked).
    pub fn progress_monitor(&self) -> ProgressMonitor {
        self.monitor.clone()
    }
    /// The progress of the running solve, or of the last one (`SolveProgress`).
    pub fn progress(&self) -> SolveProgress {
        self.monitor.snapshot()
    }
    // a change of the inputs is refused while a solve runs
    pub(crate) fn need_changeable(&self) -> Result<()> {
        if self.calculating.load(Ordering::SeqCst) {
            return Err(Error::state("the model is calculating: its inputs cannot change until the solve ends"));
        }
        Ok(())
    }
    // the results of earlier inputs are dropped once an input changed
    pub(crate) fn sync(&mut self) {
        if self.results.is_some() && self.results_revision != self.doc.revision() {
            self.results = None;
            self.results_revision = u64::MAX;
        }
    }
    /// The results of the last solve (ErrorKind::State when the model is not calculated).
    pub fn results(&self) -> Result<&Results> {
        if self.calculating.load(Ordering::SeqCst) {
            return Err(Error::state("the model is calculating: no results yet"));
        }
        match self.results.as_deref() {
            Some(r) if self.calculated() => Ok(r),
            _ => Err(Error::state("the model has no results: it was not solved, or its inputs changed since")),
        }
    }
    /// Discards the results, the inputs kept (the state is reset).
    pub fn reset(&mut self) -> Result<()> {
        self.need_changeable()?;
        self.results = None;
        self.results_revision = u64::MAX;
        Ok(())
    }
    /// Every input and the results back to a new model; the ids of its objects are refused
    /// from then on.
    pub fn clear(&mut self) -> Result<()> {
        self.need_changeable()?;
        self.results = None;
        self.results_revision = u64::MAX;
        self.doc.clear();
        Ok(())
    }

    // ---- the problem of the model

    /// The problem of the model built from its objects when they changed; the queries wait until a
    /// solve ends.
    pub(crate) fn ensure_built(&mut self) -> Result<()> {
        // the queries rebuild and recheck the problem the solve reads: they wait until it ends
        if self.calculating.load(Ordering::SeqCst) {
            return Err(Error::state("the model is calculating: it can be queried when the solve ends"));
        }
        self.build()
    }
    pub(crate) fn build(&mut self) -> Result<()> {
        if self.built_revision == self.doc.revision() {
            return Ok(());
        }
        reset_inputs(&mut self.problem);
        self.built_revision = u64::MAX;
        self.enabled = None;
        self.disabled_warnings.clear();
        // the problem of the enabled objects (the engine never sees a disabled one), its errors
        // located in the model
        let e = scoped_document(&self.doc, Scope::Model)?;
        build_scoped(&e.doc, Scope::Model, &mut self.problem).map_err(|x| e.numbering.error(x))?;
        self.enabled = Some(e);
        self.built_revision = self.doc.revision();
        Ok(())
    }
    // the numbering of the built problem in the model
    pub(crate) fn numbering(&self) -> Numbering {
        self.enabled.as_ref().map(|e| e.numbering.clone()).unwrap_or_default()
    }
    // an error of the built problem located in the model
    pub(crate) fn renumbered(&self, e: Error) -> Error {
        match &self.enabled {
            Some(x) => x.numbering.error(e),
            None => e,
        }
    }
    // the built problem with the model rules checked
    pub(crate) fn checked(&mut self) -> Result<&mut Problem> {
        self.ensure_built()?;
        if let Err(e) = self.problem.check_rules() {
            return Err(self.renumbered(e));
        }
        Ok(&mut self.problem)
    }

    // ---- the objects

    /// The document of the model (its objects, for reading).
    pub fn document(&self) -> &Document {
        &self.doc
    }
    /// The model's root object.
    pub fn root(&self) -> ObjectId {
        self.doc.root_id()
    }
    // the view of an object of the model given from outside
    pub(crate) fn obj(&self, id: ObjectId) -> Result<Obj<'_>> {
        self.doc
            .get_obj(id)
            .ok_or_else(|| Error::argument("the object was removed from its model (or the model was cleared)"))
    }
    // an object of the model whose inputs may change now
    pub(crate) fn changeable(&self, id: ObjectId) -> Result<()> {
        self.obj(id)?;
        self.need_changeable()
    }
    // a change of the document; the results of earlier inputs dropped
    fn change<T>(&mut self, id: ObjectId, f: impl FnOnce(&mut Document) -> Result<T>) -> Result<T> {
        self.changeable(id)?;
        let r = f(&mut self.doc);
        self.sync();
        r
    }
    /// The object of a path: "" the model, "options", "meshes/hull", "bodies/hull/tanks/2" (unnamed
    /// objects one-based).
    pub fn object(&self, path: &str) -> Result<ObjectId> {
        self.doc.find(path).ok_or_else(|| Error::argument(format!("no object '{path}' in the model")))
    }
    /// The type, name and place of an object.
    pub fn object_info(&self, id: ObjectId) -> Result<ObjectInfo> {
        let o = self.obj(id)?;
        Ok(ObjectInfo {
            type_name: o.type_().name,
            name: o.name().to_string(),
            list: o.list_name().to_string(),
            index: o.index(),
        })
    }
    /// The path of an object.
    pub fn path(&self, id: ObjectId) -> Result<String> {
        Ok(self.obj(id)?.path())
    }
    /// The parent of an object (None for the model).
    pub fn parent(&self, id: ObjectId) -> Result<Option<ObjectId>> {
        Ok(self.obj(id)?.parent().map(|p| p.id()))
    }
    /// The objects of a list of an object.
    pub fn children(&self, id: ObjectId, list: &str) -> Result<&[ObjectId]> {
        self.obj(id)?.try_children(list)
    }
    /// The object at a zero-based position of a list.
    pub fn child_at(&self, id: ObjectId, list: &str, index: usize) -> Result<ObjectId> {
        let o = self.obj(id)?;
        let v = o.try_children(list)?;
        v.get(index)
            .copied()
            .ok_or_else(|| Error::argument(format!("{}{}: index {} out of range", o.prefix(), list, index)))
    }
    /// Adds an object to a list of an object: of a type (None: the list's first type), with a name
    /// in a named list ("" in an unnamed one), at a zero-based position (None: at the end).
    pub fn add(
        &mut self,
        id: ObjectId,
        list: &str,
        type_: Option<&str>,
        name: &str,
        index: Option<usize>,
    ) -> Result<ObjectId> {
        let index = index.map_or(-1, |i| i64::try_from(i).unwrap_or(i64::MAX));
        let type_ = type_.unwrap_or("");
        self.change(id, |d| d.add_child(id, list, type_, name, index))
    }
    /// Removes an object and its children (their ids are refused from then on).
    pub fn remove(&mut self, id: ObjectId) -> Result<()> {
        self.change(id, |d| d.remove(id))
    }
    /// Moves an object to a zero-based position of its list.
    pub fn move_to(&mut self, id: ObjectId, index: usize) -> Result<()> {
        let index = i64::try_from(index).unwrap_or(i64::MAX);
        self.change(id, |d| d.move_(id, index))
    }
    /// Renames an object of a named list; the references to an object of the model's lists follow.
    pub fn rename(&mut self, id: ObjectId, name: &str) -> Result<()> {
        self.change(id, |d| d.rename(id, name))
    }
    /// Removes every object of a list of an object, all or nothing: refused, and nothing removed,
    /// while another object of the model refers to one of them (or to an object within them).
    pub fn clear_list(&mut self, id: ObjectId, list: &str) -> Result<()> {
        self.change(id, |d| d.clear_list(id, list))
    }
    /// The kind and state of an item.
    pub fn item_info(&self, id: ObjectId, item: &str) -> Result<ItemInfo> {
        let o = self.obj(id)?;
        let d = o.item_def(item)?;
        let v = o.get(item);
        Ok(ItemInfo {
            kind: d.kind,
            is_set: o.is_set(item),
            has_value: v.is_set(),
            is_array: v.is_array(),
            dtype: match d.kind {
                ItemKind::Array => d.dtype,
                _ => DType::Real,
            },
            shape: v.shape().iter().map(|&x| x.max(0) as usize).collect(),
            is_choice: v.is_name(),
        })
    }
    /// The value of an item: its default when not set, None when it has no value.
    pub fn get(&self, id: ObjectId, item: &str) -> Result<Option<Value>> {
        let o = self.obj(id)?;
        o.item_def(item)?;
        let v = o.get(item);
        let shape: Vec<usize> = v.shape().iter().map(|&x| x.max(0) as usize).collect();
        Ok(Some(match v {
            ItemValue::Unset => return Ok(None),
            ItemValue::Bool(b) => Value::Bool(b),
            ItemValue::Int(x) => Value::Int(x),
            ItemValue::Real(x) => Value::Real(x),
            ItemValue::Text(t) | ItemValue::Name(t) => Value::Text(t),
            ItemValue::Reals { data, .. } => Value::Reals { shape, data },
            ItemValue::Ints { data, .. } => Value::Ints { shape, data },
            ItemValue::Texts { data, .. } => Value::Texts(data),
        }))
    }
    /// Sets an item; the value is checked alone, a failure leaves the item unchanged. Setting
    /// "name" renames the object; setting a Mesh's "kind" removes its patches of another kind. An
    /// item of a surface is set only when its method (and outer) takes it; setting the method (or
    /// outer) puts the surface's other items back to their defaults.
    pub fn set(&mut self, id: ObjectId, item: &str, value: impl Into<Value>) -> Result<()> {
        let value = value.into();
        self.changeable(id)?;
        let v = self.item_value(id, item, value)?;
        self.change(id, |d| d.set(id, item, v))
    }
    // the item value of a value for an item, with the checks of its kind
    fn item_value(&self, id: ObjectId, item: &str, value: Value) -> Result<ItemValue> {
        let o = self.obj(id)?;
        let d = o.item_def(item)?;
        let scalar = |kind: ItemKind, v: ItemValue| -> Result<ItemValue> {
            if d.kind != kind {
                return Err(Error::type_(format!("{}{} is {} item", o.prefix(), item, kind_text(d.kind))));
            }
            Ok(v)
        };
        match value {
            Value::Bool(b) => scalar(ItemKind::Bool, ItemValue::Bool(b)),
            Value::Int(x) => scalar(ItemKind::Int, ItemValue::Int(x)),
            Value::Real(x) => scalar(ItemKind::Real, ItemValue::Real(x)),
            Value::Text(t) => {
                // a number item takes one of its choices as a text
                let number = (d.kind == ItemKind::Int || d.kind == ItemKind::Real) && !d.choices.is_empty();
                let texty = is_texty(d.kind) || d.kind == ItemKind::Origin || number;
                if !texty {
                    return Err(Error::type_(format!("{}{} is not a text item", o.prefix(), item)));
                }
                // kept by the model: checked as UTF-8 by set (check_value)
                Ok(if number { ItemValue::Name(t) } else { ItemValue::Text(t) })
            }
            Value::Texts(texts) => {
                if d.kind != ItemKind::Array || d.dtype != DType::Text {
                    return Err(Error::type_(format!("{}{} is not a text array", o.prefix(), item)));
                }
                let n = texts.len() as i64;
                Ok(ItemValue::Texts { shape: vec![n], data: texts })
            }
            Value::Reals { shape, data } => {
                let s = array_shape(d.kind, &o, item, &shape)?;
                Ok(ItemValue::Reals { shape: s, data })
            }
            Value::Ints { shape, data } => {
                let s = array_shape(d.kind, &o, item, &shape)?;
                Ok(ItemValue::Ints { shape: s, data })
            }
        }
    }
    /// Returns an item to its default (or to unset); the method (or outer) of a surface puts its
    /// other items back to their defaults too.
    pub fn reset_item(&mut self, id: ObjectId, item: &str) -> Result<()> {
        self.changeable(id)?;
        self.change(id, |d| d.reset(id, item))
    }
    // a body given to a function of the model: a body of it, not removed
    pub(crate) fn own_body(&self, body: ObjectId) -> Result<Obj<'_>> {
        let Some(b) = self.doc.get_obj(body) else {
            return Err(Error::argument("the body was removed from its model"));
        };
        if b.type_().name != "Body" {
            return Err(Error::argument("body: not a body of this model"));
        }
        Ok(b)
    }
}

// the shape of an array value for an item (an array, terms or an origin as a point)
fn array_shape(kind: ItemKind, o: &Obj, item: &str, shape: &[usize]) -> Result<Vec<i64>> {
    if kind != ItemKind::Array && kind != ItemKind::Terms && kind != ItemKind::Origin {
        return Err(Error::type_(format!("{}{} is not an array item", o.prefix(), item)));
    }
    if !(1..=8).contains(&shape.len()) {
        return Err(Error::value(format!(
            "{}{}: an array has 1 to 8 dimensions, got {}",
            o.prefix(),
            item,
            shape.len()
        )));
    }
    let mut size: usize = 1;
    let mut out = Vec::new();
    for &e in shape {
        if e > 0 && size > MAX_ARRAY_ELEMENTS / e {
            return Err(Error::value(format!("{}{}: more than 2^40 elements", o.prefix(), item)));
        }
        out.push(e as i64);
        size *= e;
    }
    Ok(out)
}
