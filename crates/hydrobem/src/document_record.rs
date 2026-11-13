//! The model as a binary record, the "model" entry of the model and results files: every
//! object with its type, name, items in the order they were set and lists of children. Reading
//! sets the items again through Document::set, so that every value passes the checks it passed
//! when it was given; the references (the mesh of a surface, a lid's body, a coupling's body) are set last, once every object
//! is there. The record is a byte vector.

use crate::binary::{Reader, put_f64, put_i64, put_text, put_u};
use crate::document::{DType, Document, ItemDef, ItemKind, ItemValue, MAX_ARRAY_ELEMENTS, ObjectId, is_reference_item};
use crate::error::{Error, Result};

const RECORD_VERSION: u32 = 13;

// the record of a value: its flags (an array 1, a name 2, a bool's value 4), an integer, a real and a
// text (each the value's own or zero, "" ), then for an array its element type, its shape and its
// elements
fn put_value(out: &mut Vec<u8>, v: &ItemValue) {
    put_u(out, (v.is_array() as u64) | ((v.is_name() as u64) << 1) | ((v.boolean() as u64) << 2), 1);
    put_i64(out, v.integer());
    put_f64(out, v.real());
    put_text(out, v.text());
    if !v.is_array() {
        return;
    }
    put_u(out, v.dtype() as u64, 1);
    put_u(out, v.shape().len() as u64, 1);
    for &s in v.shape() {
        put_i64(out, s);
    }
    match v {
        ItemValue::Reals { data, .. } => {
            for &x in data {
                put_f64(out, x);
            }
        }
        ItemValue::Ints { data, .. } => {
            for &x in data {
                put_i64(out, x);
            }
        }
        ItemValue::Texts { data, .. } => {
            for t in data {
                put_text(out, t);
            }
        }
        _ => {}
    }
}

// a value as its record holds it: the scalars, or an array
struct Recorded {
    name: bool,
    boolean: bool,
    integer: i64,
    real: f64,
    text: String,
    array: Option<ItemValue>,
}

impl Recorded {
    // the value of an item of the kind of d: the array, the name, or the scalar of its kind
    fn value(self, d: &ItemDef) -> ItemValue {
        if let Some(a) = self.array {
            return a;
        }
        if self.name {
            return ItemValue::Name(self.text);
        }
        match d.kind {
            ItemKind::Bool => ItemValue::Bool(self.boolean),
            ItemKind::Int => ItemValue::Int(self.integer),
            ItemKind::Real => ItemValue::Real(self.real),
            _ => ItemValue::Text(self.text),
        }
    }
}

fn read_value(r: &mut Reader) -> Result<Recorded> {
    let flags = r.u(1)?;
    let mut v = Recorded {
        name: (flags & 2) != 0,
        boolean: (flags & 4) != 0,
        integer: r.i64()?,
        real: r.f64()?,
        text: r.text()?,
        array: None,
    };
    if (flags & 1) == 0 {
        return Ok(v);
    }
    let dtype = r.u(1)?;
    let dtype = match dtype {
        0 => DType::Real,
        1 => DType::Int,
        2 => DType::Text,
        _ => return Err(Error::value(format!("{}: an item has an unknown element type", r.where_()))),
    };
    let rank = r.u(1)?;
    if rank > 8 {
        return Err(Error::value(format!("{}: an item has more than eight dimensions", r.where_())));
    }
    let mut n: usize = 1;
    let mut shape = Vec::new();
    for _ in 0..rank {
        let s = r.i64()?;
        if s < 0 || (s > 0 && n > MAX_ARRAY_ELEMENTS / s as usize) {
            return Err(Error::value(format!("{}: an item has an invalid shape", r.where_())));
        }
        shape.push(s);
        n *= s as usize;
    }
    // the elements must be in the record before anything is allocated for them (8 bytes each, a
    // text at least its 4-byte length): a damaged count is refused, not allocated
    if n > r.left() / (if dtype == DType::Text { 4 } else { 8 }) {
        return Err(Error::value(format!("{}: the record is damaged (cut short)", r.where_())));
    }
    v.array = Some(match dtype {
        DType::Real => {
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(r.f64()?);
            }
            ItemValue::Reals { shape, data }
        }
        DType::Int => {
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(r.i64()?);
            }
            ItemValue::Ints { shape, data }
        }
        DType::Text => {
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(r.text()?);
            }
            ItemValue::Texts { shape, data }
        }
    });
    Ok(v)
}

type Refs = Vec<(ObjectId, String, ItemValue)>;

impl Document {
    /// The model as a binary record: every object, its items in the order they were set and
    /// its children.
    pub fn record(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u(&mut out, RECORD_VERSION as u64, 4);
        self.put_object(&mut out, self.root_id());
        out
    }
    /// Reads a record into this new document, setting every item again (its checks run); where_
    /// names the record in the messages.
    pub fn load_record(&mut self, bytes: &[u8], where_: &str) -> Result<()> {
        let mut r = Reader::new(bytes, where_.to_string());
        let version = r.u(4)?;
        if version != RECORD_VERSION as u64 {
            return Err(Error::value(format!(
                "{where_}: model record version {version}; this HydroBEM reads version {RECORD_VERSION} (no conversion)"
            )));
        }
        let mut refs: Refs = Vec::new();
        let root = self.root_id();
        let read = (|| -> Result<()> {
            let type_ = r.text()?;
            let name = r.text()?;
            read_body(&mut r, self, root, &mut refs, &type_, &name)?;
            for (o, item, v) in refs.drain(..) {
                self.set(o, &item, v)?;
            }
            Ok(())
        })();
        if let Err(e) = read {
            // a value the checks refuse is a damaged (or edited) file: named with the file
            let m = e.message();
            let message = if m.starts_with(where_) { m.to_string() } else { format!("{where_}: {m}") };
            return Err(Error::value(message).at(*e.location()));
        }
        if !r.done() {
            return Err(Error::value(format!("{where_}: the model record has data after its end")));
        }
        Ok(())
    }

    fn put_object(&self, out: &mut Vec<u8>, id: ObjectId) {
        let o = self.node(id);
        put_text(out, o.type_.name);
        put_text(out, &o.name);
        // the items in the order they were set, the references last (as reading sets them), so that a
        // model read and written again gives the same record
        put_u(out, o.items.len() as u64, 4);
        for refs in 0..2 {
            for (name, value) in &o.items {
                let is_ref = is_reference_item(o.type_.item(name).expect("a set item is in the schema"));
                if is_ref == (refs == 1) {
                    put_text(out, name);
                    put_value(out, value);
                }
            }
        }
        put_u(out, o.lists.len() as u64, 4);
        for (list, children) in &o.lists {
            put_text(out, list);
            put_u(out, children.len() as u64, 4);
            for &c in children {
                self.put_object(out, c);
            }
        }
    }
}

// The record of an object after its type and name, which the record of its parent read to add it
// (read once and handed on: the same bytes in the same order).
fn read_body(r: &mut Reader, doc: &mut Document, id: ObjectId, refs: &mut Refs, type_: &str, name: &str) -> Result<()> {
    let t = doc.obj(id).type_();
    if type_ != t.name {
        return Err(Error::value(format!("{}: a {} where the model has a {}", r.where_(), type_, t.name)));
    }
    let _ = name; // given when the object was added
    let items = r.u(4)?;
    for _ in 0..items {
        let item = r.text()?;
        let v = read_value(r)?;
        let d = doc.obj(id).item_def(&item)?;
        let v = v.value(d);
        if is_reference_item(d) {
            refs.push((id, item, v));
        } else {
            doc.set(id, &item, v)?;
        }
    }
    let lists = r.u(4)?;
    for _ in 0..lists {
        let list = r.text()?;
        let Some(l) = t.list(&list) else {
            return Err(Error::value(format!(
                "{}: {}no list '{}' in a {}",
                r.where_(),
                doc.obj(id).prefix(),
                list,
                t.name
            )));
        };
        let n = r.u(4)?;
        if l.always && n != 1 {
            return Err(Error::value(format!("{}: {} must hold one object, not {}", r.where_(), list, n)));
        }
        for _ in 0..n {
            // the type and name of the child come first in its record
            let child_type = r.text()?;
            let child_name = r.text()?;
            let c = if l.always {
                doc.obj(id).children_ids(&list)[0]
            } else {
                doc.add_child(id, &list, &child_type, &child_name, -1)?
            };
            read_body(r, doc, c, refs, &child_type, &child_name)?;
        }
    }
    Ok(())
}
