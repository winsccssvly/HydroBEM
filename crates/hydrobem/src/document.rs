//! The model as objects with named data items: the object tree, the item values and
//! their checks, the schema, and the model as a binary record (the model and results files).
//! The schema is in document_schema.rs, the model record in document_record.rs (impl blocks of the
//! types here). This file: values, the checks of a value alone, the object tree and its paths.
//!
//! The object tree is an arena: the Document owns the objects in slots, an object is addressed by
//! an `ObjectId` (its slot and the generation of the slot), and its parent and children are ids.
//! An object taken out of the model (removed, replaced or cleared) frees its slot and the slot's
//! generation moves on, so that an id kept from before is refused (`Document::contains`) rather
//! than taken for an object that later reuses the slot. `Obj` is a borrowed view of one
//! object for reading, `ObjMut` one for changing it; the changes are methods of the Document
//! that take ids.
//!
//! Text kept in a model is UTF-8. A `String` is UTF-8 already; bytes from outside (a C
//! string) are checked first with `check_utf8` (or `Obj::item_text`), which gives the message of
//! the check of the model. The NUL check stays where the model takes a text.

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex};

use crate::constants::STANDARD_GRAVITY;
use crate::document_rules::ItemRule;
use crate::error::{Error, Result};
use crate::mesh::build_mesh;
use crate::mesh_desc::{LowOrderMeshDesc, MeshKind};
use crate::text::{py_repr, py_str_repr, strip};

pub use crate::document_schema::{schema, schema_type};

/// The kind of a data item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ItemKind {
    #[default]
    Bool,
    Int,
    Real,
    Text,
    Choice,
    Array,
    Reference,
    Origin,
    Terms,
}

impl ItemKind {
    /// Its name in the schema, e.g. "reference".
    pub fn name(self) -> &'static str {
        match self {
            ItemKind::Bool => "bool",
            ItemKind::Int => "int",
            ItemKind::Real => "real",
            ItemKind::Text => "text",
            ItemKind::Choice => "choice",
            ItemKind::Array => "array",
            ItemKind::Reference => "reference",
            ItemKind::Origin => "origin",
            ItemKind::Terms => "terms",
        }
    }
}

/// A reference to an object of the root: it follows a rename and keeps the object it names.
pub fn is_reference(k: ItemKind) -> bool {
    k == ItemKind::Reference
}

/// Whether an item names an object of the model: a reference (its target a list of the root:
/// "meshes", "bodies"; the object by its name).
pub fn is_reference_item(d: &ItemDef) -> bool {
    is_reference(d.kind)
}

// whether a reference of target t and value v names the object at the address (p, a); a target is
// a list of the root, so a reference names no object within another
fn names_within(t: &str, v: &str, p: &str, a: &str) -> bool {
    t == p && v == a
}

// whether a reference of target t names an object of the list p of the object at the address a
// ("" the root): a list of the root, its target
fn names_into(t: &str, p: &str, a: &str) -> bool {
    t == p && a.is_empty()
}

/// The address of an object for the references: its lists from the root ("bodies/tanks") and its
/// names ("spar/t1"); None for the root and for an object without a name (or with an unnamed
/// object on its way).
pub fn address(o: Obj) -> Option<(String, String)> {
    let parent = o.parent()?;
    if o.name().is_empty() {
        return None;
    }
    if parent.parent().is_none() {
        return Some((o.list_name().to_string(), o.name().to_string()));
    }
    let (p, a) = address(parent)?;
    Some((format!("{p}/{}", o.list_name()), format!("{a}/{}", o.name())))
}

/// The object a reference names: target a list of the root, v the name of its object; the message
/// of the failure otherwise (an unknown name).
pub fn resolve_reference(root: Obj<'_>, target: &str, v: &str) -> std::result::Result<ObjectId, String> {
    let what = if target == "meshes" { "mesh" } else { singular(target) };
    match root.child(target, v).filter(|_| !v.is_empty()) {
        Some(c) => Ok(c.id),
        None => Err(format!("unknown {what} {}", py_str_repr(v))),
    }
}

// a reference as the path of the object it names ("meshes/hull")
fn reference_path(target: &str, v: &str) -> String {
    format!("{target}/{v}")
}
/// The element type of an array item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DType {
    #[default]
    Real,
    Int,
    Text,
}

impl DType {
    /// Its name in the schema: "real", "int" or "text".
    pub fn name(self) -> &'static str {
        match self {
            DType::Real => "real",
            DType::Int => "int",
            DType::Text => "text",
        }
    }
}

/// The largest number of elements of an array item: a bound for the products of the extents,
/// which could otherwise wrap.
pub const MAX_ARRAY_ELEMENTS: usize = 1usize << 40;

/// The value of an item: unset, a scalar of the item's kind, one of the names a number item may
/// take, or an array (C order).
#[derive(Clone, Debug, Default)]
pub enum ItemValue {
    /// No value (an unset item without a default).
    #[default]
    Unset,
    Bool(bool),
    Int(i64),
    Real(f64),
    /// A Text, Choice or Reference item, or an Origin given by name.
    Text(String),
    /// An Int or Real item given as one of its names ("auto", "displaced").
    Name(String),
    /// An array of reals: an Array of reals, a Terms item, an Origin given as a point.
    Reals {
        shape: Vec<i64>,
        data: Vec<f64>,
    },
    /// An array of integers.
    Ints {
        shape: Vec<i64>,
        data: Vec<i64>,
    },
    /// An array of texts.
    Texts {
        shape: Vec<i64>,
        data: Vec<String>,
    },
}

// the views of a value: the part its variant holds, or the empty value (false, 0, "", no elements)
impl ItemValue {
    /// Whether it is a value (not `Unset`).
    pub fn is_set(&self) -> bool {
        !matches!(self, ItemValue::Unset)
    }
    /// Whether it is an array (`Reals`, `Ints` or `Texts`).
    pub fn is_array(&self) -> bool {
        matches!(self, ItemValue::Reals { .. } | ItemValue::Ints { .. } | ItemValue::Texts { .. })
    }
    /// Whether it is a number item given as one of its names (`Name`).
    pub fn is_name(&self) -> bool {
        matches!(self, ItemValue::Name(_))
    }
    /// The element type of an array (`Real` for a value that is not one).
    pub fn dtype(&self) -> DType {
        match self {
            ItemValue::Ints { .. } => DType::Int,
            ItemValue::Texts { .. } => DType::Text,
            _ => DType::Real,
        }
    }
    /// A `Bool`'s value, false otherwise.
    pub fn boolean(&self) -> bool {
        matches!(self, ItemValue::Bool(true))
    }
    /// An `Int`'s value, 0 otherwise.
    pub fn integer(&self) -> i64 {
        if let ItemValue::Int(x) = self { *x } else { 0 }
    }
    /// A `Real`'s value, 0.0 otherwise.
    pub fn real(&self) -> f64 {
        if let ItemValue::Real(x) = self { *x } else { 0.0 }
    }
    /// The text of a `Text` or a `Name`, "" otherwise.
    pub fn text(&self) -> &str {
        match self {
            ItemValue::Text(s) | ItemValue::Name(s) => s,
            _ => "",
        }
    }
    /// The shape of an array, none otherwise.
    pub fn shape(&self) -> &[i64] {
        match self {
            ItemValue::Reals { shape, .. } | ItemValue::Ints { shape, .. } | ItemValue::Texts { shape, .. } => shape,
            _ => &[],
        }
    }
    /// The elements of `Reals`, none otherwise.
    pub fn reals(&self) -> &[f64] {
        if let ItemValue::Reals { data, .. } = self { data } else { &[] }
    }
    /// The elements of `Ints`, none otherwise.
    pub fn ints(&self) -> &[i64] {
        if let ItemValue::Ints { data, .. } = self { data } else { &[] }
    }
    /// The elements of `Texts`, none otherwise.
    pub fn texts(&self) -> &[String] {
        if let ItemValue::Texts { data, .. } = self { data } else { &[] }
    }
    // the shape of an array, for the check that completes an empty one
    fn shape_mut(&mut self) -> Option<&mut Vec<i64>> {
        match self {
            ItemValue::Reals { shape, .. } | ItemValue::Ints { shape, .. } | ItemValue::Texts { shape, .. } => {
                Some(shape)
            }
            _ => None,
        }
    }
}

impl PartialEq for ItemValue {
    fn eq(&self, o: &ItemValue) -> bool {
        use ItemValue as V;
        match (self, o) {
            (V::Unset, V::Unset) => true,
            (V::Bool(a), V::Bool(b)) => a == b,
            (V::Int(a), V::Int(b)) => a == b,
            // bit for bit (NaN equal to itself)
            (V::Real(a), V::Real(b)) => a.to_bits() == b.to_bits(),
            (V::Text(a), V::Text(b)) | (V::Name(a), V::Name(b)) => a == b,
            (V::Reals { shape: s, data: a }, V::Reals { shape: t, data: b }) => {
                s == t && a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (V::Ints { shape: s, data: a }, V::Ints { shape: t, data: b }) => s == t && a == b,
            (V::Texts { shape: s, data: a }, V::Texts { shape: t, data: b }) => s == t && a == b,
            _ => false,
        }
    }
}

/// A list of child objects: named or numbered, or one optional child.
#[derive(Clone, Debug, Default)]
pub struct ListDef {
    pub name: &'static str,
    pub types: Vec<&'static str>,  // the object types it holds
    pub named: bool,               // every object has a name, unique within the list
    pub optional_names: bool,      // an object may have a name (unique within the list), or none
    pub single: bool,              // at most one child (always present for the model's own)
    pub always: bool,              // single and created with its parent
    pub description: &'static str, // what the list holds
}

/// One data item of an object type.
#[derive(Clone, Debug, Default)]
pub struct ItemDef {
    pub name: &'static str,
    pub kind: ItemKind,
    pub dtype: DType,                   // arrays
    pub shape: &'static str,            // arrays: the shape rule, e.g. "3", "N,3", "P,4,3", "6,6N", "3|3,3"
    pub unit: &'static str,             // SI unit, "" for none
    pub fallback: Option<&'static str>, // the default as a schema literal ("x", 1.5, [0.0, 0.0]); None: unset
    pub choices: Vec<&'static str>, // Choice, Origin, text arrays; Int, Real: the names it may take instead of a number
    pub rule: ItemRule,             // the check of the value alone (document.rs; document_rules.rs)
    pub lo: f64,                    // the range of rules that take one
    pub hi: f64,
    pub target: &'static str, // Reference: the list of the root it names ("meshes", "bodies")
    pub kinds: &'static str,  // Mesh: the mesh kinds that have this item ("" every kind)
    pub switch: &'static str, // the switch items that choose this item, separated by spaces ("": none)
    pub switch_values: &'static str, // the values of the switches that take it, separated by spaces,
    // each "v" (the first switch) or "v/w" (the first and the second): "cylinder automatic/circle"
    pub description: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct TypeDef {
    pub name: &'static str,
    pub items: Vec<ItemDef>,
    pub lists: Vec<ListDef>,
}
// TypeDef::item and TypeDef::list are in document_schema.rs.

// ------------------------------------------------------------------ shape rules

fn split(s: &str, sep: char) -> Vec<String> {
    s.split(sep).map(String::from).collect()
}

/// Whether bytes are UTF-8 as a model keeps its text (the model file holds it as UTF-8):
/// well-formed sequences of 1 to 4 bytes, no overlong forms, surrogates or code points above
/// U+10FFFF (RFC 3629).
pub fn valid_utf8(s: &[u8]) -> bool {
    let mut i = 0usize;
    while i < s.len() {
        let c = s[i];
        let (n, mut cp): (usize, u32) = if c < 0x80 {
            (0, c as u32)
        } else if (c & 0xe0) == 0xc0 {
            (1, (c & 0x1f) as u32)
        } else if (c & 0xf0) == 0xe0 {
            (2, (c & 0x0f) as u32)
        } else if (c & 0xf8) == 0xf0 {
            (3, (c & 0x07) as u32)
        } else {
            return false;
        };
        if i + n >= s.len() {
            return false; // the sequence is cut off
        }
        for k in 1..=n {
            let t = s[i + k];
            if (t & 0xc0) != 0x80 {
                return false;
            }
            cp = (cp << 6) | (t & 0x3f) as u32;
        }
        const LEAST: [u32; 4] = [0, 0x80, 0x800, 0x10000];
        if cp < LEAST[n] || cp > 0x10ffff || (0xd800..=0xdfff).contains(&cp) {
            return false;
        }
        i += n + 1;
    }
    true
}

/// The check of a text of the model: ErrorKind::Value "<where_>the text is not valid UTF-8", or
/// "<where_>the text has a NUL character"; the text as a str. Bytes from outside (a C string) are
/// checked with it before they become a String, with the where_ of the place the model takes the
/// text: `<object prefix><item>: ` for an item value (`Obj::item_text`), `<object prefix>` for a
/// new name (rename), `<object prefix><list>: ` for the name of a new child (add).
pub fn check_utf8<'a>(where_: &str, s: &'a [u8]) -> Result<&'a str> {
    if !valid_utf8(s) {
        return Err(Error::value(format!("{where_}the text is not valid UTF-8")));
    }
    if s.contains(&0) {
        return Err(Error::value(format!("{where_}the text has a NUL character"))); // a C string ends there
    }
    match std::str::from_utf8(s) {
        Ok(t) => Ok(t),
        Err(_) => Err(Error::internal(format!("{where_}the text is not valid UTF-8"))),
    }
}

// A name of a named list: not blank, and without '/' (the separator of the object paths) or
// control characters
fn check_name(where_: &str, name: &str) -> Result<()> {
    check_utf8(where_, name.as_bytes())?;
    if strip(name).is_empty() {
        return Err(Error::value(format!("{where_}names must not be blank")));
    }
    for c in name.bytes() {
        if c == b'/' || c < 0x20 || c == 0x7f {
            return Err(Error::value(format!(
                "{where_}the name {} has a '/' or a control character",
                py_str_repr(name)
            )));
        }
    }
    Ok(())
}

// a name of a list with optional names: not digits alone, which address an object by its position
// in the paths ("bodies/spar/dipoles/2")
fn check_optional_name(where_: &str, l: &ListDef, name: &str) -> Result<()> {
    if l.optional_names && all_digits(name) {
        return Err(Error::value(format!(
            "{where_}the name {} is a position: a name of {} needs a character other than a digit",
            py_str_repr(name),
            l.name
        )));
    }
    Ok(())
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())
}

// whether a shape matches one alternative of a rule ("6,6N", "M,M", "P,4,3")
fn matches(shape: &[i64], alt: &str) -> bool {
    let dims = split(alt, ',');
    if dims.len() != shape.len() {
        return false;
    }
    let mut vars: BTreeMap<String, i64> = BTreeMap::new();
    for (k, d) in dims.iter().enumerate() {
        if all_digits(d) {
            if shape[k] != d.parse::<i64>().unwrap_or(-1) {
                return false;
            }
        } else if d == "6N" {
            if shape[k] <= 0 || shape[k] % 6 != 0 {
                return false;
            }
        } else {
            let e = *vars.entry(d.clone()).or_insert(shape[k]);
            if e != shape[k] {
                return false;
            }
        }
    }
    true
}

// the shape of an empty value under the first alternative (the variables 0)
fn empty_shape(d: &ItemDef) -> Vec<i64> {
    let mut s: Vec<i64> = Vec::new();
    for x in split(&split(d.shape, '|')[0], ',') {
        s.push(if all_digits(&x) { x.parse::<i64>().unwrap_or(0) } else { 0 });
    }
    s[0] = 0;
    s
}

fn shape_of(s: &[i64]) -> String {
    let mut t = String::from("(");
    for (k, x) in s.iter().enumerate() {
        if k > 0 {
            t += ", ";
        }
        t += &x.to_string();
    }
    t + if s.len() == 1 { ",)" } else { ")" }
}

/// The shape rule of an array item as text ("(3,)", "(N, 3)") for the messages.
pub fn shape_text(d: &ItemDef) -> String {
    let mut out = String::new();
    for alt in split(d.shape, '|') {
        let dims = split(&alt, ',');
        let mut t = String::from("(");
        for (k, x) in dims.iter().enumerate() {
            if k > 0 {
                t += ", ";
            }
            t += x;
        }
        t += if dims.len() == 1 { ",)" } else { ")" };
        if !out.is_empty() {
            out += " or ";
        }
        out += &t;
    }
    out
}

// ------------------------------------------------------------------ literals <-> values

fn kind_name(k: ItemKind) -> &'static str {
    match k {
        ItemKind::Bool => "a bool",
        ItemKind::Int => "an integer",
        ItemKind::Real => "a real number",
        ItemKind::Text => "a string",
        ItemKind::Choice => "a string",
        ItemKind::Reference => "a name",
        _ => "an array",
    }
}

// ------------------------------------------------------------------ the literals of the schema

// A default of the schema: a quoted string, true or false, a number (inf, nan), or [ ... ]
// of these; the notation of the defaults in document_schema.rs, read once per item.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LitKind {
    Boolean,
    Integer,
    Float,
    String,
    Array,
}

struct Literal {
    kind: LitKind,
    boolean: bool,
    integer: i64,
    number: f64,
    text: String,
    items: Vec<Literal>,
}

impl Literal {
    fn new() -> Literal {
        Literal {
            kind: LitKind::String,
            boolean: false,
            integer: 0,
            number: 0.0,
            text: String::new(),
            items: Vec::new(),
        }
    }
    fn is_array(&self) -> bool {
        self.kind == LitKind::Array
    }
}

fn read_literal(s: &str, at: &mut usize) -> Result<Literal> {
    let b = s.as_bytes();
    let skip = |at: &mut usize| {
        while *at < b.len() && (b[*at] == b' ' || b[*at] == b',') {
            *at += 1;
        }
    };
    skip(at);
    if *at >= b.len() {
        return Err(Error::internal(format!("schema default '{s}': a value is missing")));
    }
    let mut v = Literal::new();
    if b[*at] == b'[' {
        v.kind = LitKind::Array;
        *at += 1;
        skip(at);
        while *at < b.len() && b[*at] != b']' {
            v.items.push(read_literal(s, at)?);
            skip(at);
        }
        if *at >= b.len() {
            return Err(Error::internal(format!("schema default '{s}': no closing ]")));
        }
        *at += 1;
        return Ok(v);
    }
    if b[*at] == b'"' {
        let Some(end) = s[*at + 1..].find('"').map(|x| x + *at + 1) else {
            return Err(Error::internal(format!("schema default '{s}': no closing quote")));
        };
        v.text = s[*at + 1..end].to_string();
        *at = end + 1;
        return Ok(v);
    }
    let mut end = *at;
    while end < b.len() && b[end] != b',' && b[end] != b']' && b[end] != b' ' {
        end += 1;
    }
    let w = &s[*at..end];
    *at = end;
    if w == "true" || w == "false" {
        v.kind = LitKind::Boolean;
        v.boolean = w == "true";
    } else if !w.contains(['.', 'e', 'E', 'n', 'i']) {
        v.kind = LitKind::Integer;
        v.integer = w.parse::<i64>().map_err(|_| Error::internal(format!("schema default '{s}': not a number")))?;
    } else {
        v.kind = LitKind::Float;
        v.number = match w {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            _ => w.parse::<f64>().map_err(|_| Error::internal(format!("schema default '{s}': not a number")))?,
        };
    }
    Ok(v)
}

// a rectangular nested array of numbers or strings; false when it is not one
// the leaves of an array literal by element type
#[derive(Default)]
struct Leaves {
    reals: Vec<f64>,
    ints: Vec<i64>,
    texts: Vec<String>,
}

fn flatten(v: &Literal, t: DType, shape: &mut Vec<i64>, depth: usize, out: &mut Leaves) -> bool {
    if v.is_array() {
        if shape.len() == depth {
            shape.push(v.items.len() as i64);
        } else if shape[depth] != v.items.len() as i64 {
            return false;
        }
        for x in &v.items {
            if !flatten(x, t, shape, depth + 1, out) {
                return false;
            }
        }
        return true;
    }
    if depth != shape.len() {
        return false; // a scalar where the others are arrays
    }
    if t == DType::Text {
        if v.kind != LitKind::String {
            return false;
        }
        out.texts.push(v.text.clone());
    } else if t == DType::Int {
        if v.kind != LitKind::Integer {
            return false;
        }
        out.ints.push(v.integer);
    } else if v.kind == LitKind::Float {
        out.reals.push(v.number);
    } else if v.kind == LitKind::Integer {
        out.reals.push(v.integer as f64);
    } else {
        return false;
    }
    true
}

fn array_value(d: &ItemDef, v: &Literal, t: DType) -> Result<ItemValue> {
    let mut out = Leaves::default();
    let mut shape: Vec<i64> = Vec::new();
    if !v.is_array() || !flatten(v, t, &mut shape, 0, &mut out) {
        let what = match t {
            DType::Text => "strings",
            DType::Int => "integers",
            DType::Real => "real numbers",
        };
        return Err(Error::type_(format!("{} must be an array of {} of shape {}", d.name, what, shape_text(d))));
    }
    let mut n: usize = 1;
    for &x in &shape {
        n *= x as usize;
    }
    let shape = if n == 0 { empty_shape(d) } else { shape };
    Ok(match t {
        DType::Real => ItemValue::Reals { shape, data: out.reals },
        DType::Int => ItemValue::Ints { shape, data: out.ints },
        DType::Text => ItemValue::Texts { shape, data: out.texts },
    })
}

fn terms_value(d: &ItemDef, v: &Literal) -> Result<ItemValue> {
    let mut data = Vec::new();
    let what = format!("{} must be a sequence of (coefficient, (px, py, pz)) terms", d.name);
    if !v.is_array() {
        return Err(Error::type_(what));
    }
    for t in &v.items {
        if !t.is_array() || t.items.len() != 2 {
            return Err(Error::type_(format!("{}: a term is (coefficient, (px, py, pz))", d.name)));
        }
        let c = &t.items[0];
        let p = &t.items[1];
        if c.kind != LitKind::Float && c.kind != LitKind::Integer {
            return Err(Error::type_(format!("{} coefficient must be a real number", d.name)));
        }
        if !p.is_array() || p.items.len() != 3 {
            return Err(Error::type_(format!("{}: the powers of a term are (px, py, pz)", d.name)));
        }
        data.push(if c.kind == LitKind::Float { c.number } else { c.integer as f64 });
        for k in &p.items {
            if k.kind != LitKind::Integer {
                return Err(Error::type_(format!("{}: the powers must be integers", d.name)));
            }
            data.push(k.integer as f64);
        }
    }
    Ok(ItemValue::Reals { shape: vec![v.items.len() as i64, 4], data })
}

// the value of a literal for an item, before the checks of the rules
// "a real number or one of displaced"
fn number_or_choice(d: &ItemDef) -> String {
    let mut out = String::from(if d.kind == ItemKind::Int { "an integer" } else { "a real number" });
    if d.choices.is_empty() {
        return out;
    }
    out += " or one of ";
    out += &d.choices.join(", ");
    out
}

fn from_literal(d: &ItemDef, v: &Literal) -> Result<ItemValue> {
    match d.kind {
        ItemKind::Bool => {
            if v.kind != LitKind::Boolean {
                return Err(Error::type_(format!("{} must be a bool", d.name)));
            }
            Ok(ItemValue::Bool(v.boolean))
        }
        ItemKind::Int => {
            if v.kind == LitKind::String && !d.choices.is_empty() {
                return Ok(ItemValue::Name(v.text.clone()));
            }
            if v.kind != LitKind::Integer {
                return Err(Error::type_(format!("{} must be {}", d.name, number_or_choice(d))));
            }
            Ok(ItemValue::Int(v.integer))
        }
        ItemKind::Real => {
            if v.kind == LitKind::String && !d.choices.is_empty() {
                return Ok(ItemValue::Name(v.text.clone()));
            }
            if v.kind == LitKind::Float {
                Ok(ItemValue::Real(v.number))
            } else if v.kind == LitKind::Integer {
                Ok(ItemValue::Real(v.integer as f64))
            } else {
                Err(Error::type_(format!("{} must be {}", d.name, number_or_choice(d))))
            }
        }
        ItemKind::Text | ItemKind::Choice | ItemKind::Reference => {
            if v.kind != LitKind::String {
                return Err(Error::type_(format!("{} must be {}", d.name, kind_name(d.kind))));
            }
            Ok(ItemValue::Text(v.text.clone()))
        }
        ItemKind::Origin => {
            if v.kind == LitKind::String {
                return Ok(ItemValue::Text(v.text.clone()));
            }
            if !v.is_array() || v.items.len() != 3 {
                return Err(Error::type_(format!(
                    "{} must be one of {}, {} or a point (x, y, z)",
                    d.name, d.choices[0], d.choices[1]
                )));
            }
            array_value(d, v, DType::Real)
        }
        ItemKind::Array => array_value(d, v, d.dtype),
        ItemKind::Terms => terms_value(d, v),
    }
}

// ------------------------------------------------------------------ the checks of a value alone

fn got(x: f64) -> String {
    py_repr(x)
}

// the elements of a value as doubles (reals, or integers)
fn numbers(v: &ItemValue) -> Vec<f64> {
    if !v.is_array() {
        return vec![v.real()];
    }
    if v.dtype() == DType::Int {
        return v.ints().iter().map(|&k| k as f64).collect();
    }
    v.reals().to_vec()
}

#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn check_rule(d: &ItemDef, v: &ItemValue) -> Result<()> {
    let n = d.name;
    let integer = d.kind == ItemKind::Int || (v.is_array() && v.dtype() == DType::Int);
    match d.rule {
        ItemRule::None | ItemRule::Real => {}
        ItemRule::Finite => {
            for x in numbers(v) {
                if !x.is_finite() {
                    return Err(Error::value(format!("{n} must be finite")));
                }
            }
        }
        ItemRule::Positive => {
            for x in numbers(v) {
                if !(x.is_finite() && x > 0.0) {
                    return Err(Error::value(format!("{n} must be a positive finite number, got {}", got(x))));
                }
            }
        }
        ItemRule::Depth => {
            if v.real().is_nan() || !(v.real() > 0.0) {
                return Err(Error::value(format!("{n} must be > 0 or inf, got {}", got(v.real()))));
            }
        }
        ItemRule::Nonnegative => {
            for x in numbers(v) {
                if !(x.is_finite() && x >= 0.0) {
                    return Err(Error::value(if integer {
                        format!("{n} must be >= 0")
                    } else {
                        format!("{n} must be a finite number >= 0, got {}", got(x))
                    }));
                }
            }
        }
        ItemRule::Unit => {
            if !(v.real().is_finite() && v.real() >= 0.0 && v.real() <= 1.0) {
                return Err(Error::value(format!("{n} must be in 0..1")));
            }
        }
        ItemRule::Float32 => {
            for x in numbers(v) {
                if !x.is_finite() || x.abs() > f32::MAX as f64 {
                    return Err(Error::value(format!("{n} must contain finite real numbers that fit float32")));
                }
            }
        }
        ItemRule::IntRange => {
            if (v.integer() as f64) < d.lo || (v.integer() as f64) > d.hi {
                return Err(Error::value(format!("{n} must be in {}..{}", d.lo as i64, d.hi as i64)));
            }
        }
        ItemRule::Name => {
            if strip(v.text()).is_empty() {
                return Err(Error::value(format!("{n} must not be blank")));
            }
        }
        ItemRule::Modes => {
            const MODES: [&str; 6] = ["surge", "sway", "heave", "roll", "pitch", "yaw"];
            let mut last: i32 = -1;
            for m in v.texts() {
                let Some(k) = MODES.iter().position(|x| *x == m) else {
                    return Err(Error::value(format!(
                        "{n}: unknown mode {} (use surge, sway, heave, roll, pitch, yaw)",
                        py_str_repr(m)
                    )));
                };
                let k = k as i32;
                if k <= last {
                    return Err(Error::value(format!(
                        "{n} must be unique and in the order surge, sway, heave, roll, pitch, yaw"
                    )));
                }
                last = k;
            }
        }
        ItemRule::Ranges => {} // check_ranges
        ItemRule::Omegas => {
            for &x in v.reals() {
                if x.is_nan() || x < 0.0 {
                    return Err(Error::value(format!("{n} must be >= 0 (0 and +inf request the limits)")));
                }
            }
        }
        ItemRule::Points => {
            if v.shape()[0] < 1 {
                return Err(Error::value(format!("{n} must be an (N, 3) array with N >= 1")));
            }
            for &x in v.reals() {
                if !x.is_finite() {
                    return Err(Error::value(format!("{n} must be finite")));
                }
            }
        }
        ItemRule::Gyration => {
            for &x in v.reals() {
                if !x.is_finite() {
                    return Err(Error::value(format!("{n} must be finite")));
                }
            }
            if v.shape().len() == 2 {
                for a in 0..3 {
                    for b in 0..3 {
                        if v.reals()[3 * a + b] != v.reals()[3 * b + a] {
                            return Err(Error::value(format!("{n} must be a symmetric matrix")));
                        }
                    }
                }
            }
        }
        ItemRule::Discretization => {
            let mut k = 0usize;
            while k + 6 <= v.ints().len() {
                let x = &v.ints()[k..k + 6];
                if x[0] < 1 || x[1] < 1 {
                    return Err(Error::value(
                        "the potential spans nu, nv (the columns 1, 2 of discretization) must be >= 1",
                    ));
                }
                if x[2] < 2 || x[2] > 6 || x[3] < 2 || x[3] > 6 {
                    return Err(Error::value(
                        "the potential orders ku, kv (the columns 3, 4 of discretization) must be in 2..6",
                    ));
                }
                if x[4] < 1 || x[5] < 1 {
                    return Err(Error::value(
                        "the Gauss orders iquo, iqui (the columns 5, 6 of discretization) must be >= 1",
                    ));
                }
                k += 6;
            }
        }
        ItemRule::Terms => {
            let mut k = 0usize;
            while k < v.reals().len() {
                if !v.reals()[k].is_finite() {
                    return Err(Error::value(format!("{n} coefficient must be finite")));
                }
                for j in 1..4 {
                    let p = v.reals()[k + j];
                    if p != p.floor() {
                        return Err(Error::value(format!("{n}: the powers must be integers")));
                    }
                    if !(0.0..=8.0).contains(&p) {
                        return Err(Error::value(format!("{n}: the powers must be in 0..8")));
                    }
                }
                k += 4;
            }
        }
    }
    Ok(())
}

/// The symmetry planes of a mesh ("none", "yz", "xz", "xz_and_yz") as the two symmetry flags of a
/// mesh file (x, y): x the mesh reflected about the plane x = 0 (yz), y about y = 0 (xz).
pub fn symmetry_flags(v: &ItemValue) -> [i32; 2] {
    let t = v.text();
    [(t == "yz" || t == "xz_and_yz") as i32, (t == "xz" || t == "xz_and_yz") as i32]
}
/// The item value of the two symmetry flags (x, y) of a mesh file.
pub fn symmetry_value(isx: i32, isy: i32) -> ItemValue {
    const NAMES: [[&str; 2]; 2] = [["none", "xz"], ["yz", "xz_and_yz"]];
    ItemValue::Text(NAMES[(isx != 0) as usize][(isy != 0) as usize].to_string())
}

/// The checks of a value alone (the kind, the shape, the rule); element ranges are kept sorted.
pub fn check_value(d: &ItemDef, v: &mut ItemValue) -> Result<()> {
    if !v.is_set() {
        return Ok(());
    }
    let at = format!("{}: ", d.name);
    check_utf8(&at, v.text().as_bytes())?;
    for t in v.texts() {
        check_utf8(&at, t.as_bytes())?;
    }
    let number = d.kind == ItemKind::Int || d.kind == ItemKind::Real;
    if v.is_name() && !(number && !d.choices.is_empty()) {
        return Err(Error::type_(format!("{} must be {}", d.name, kind_name(d.kind))));
    }
    // a scalar holds the variant of its item's kind, so that the typed reads, Model::get and the
    // record see that kind: an integer given to a real item is taken as a real (as in an array),
    // another variant is refused (a name only for a number item with choices, above)
    if !v.is_array() && !v.is_name() {
        if d.kind == ItemKind::Real && matches!(v, ItemValue::Int(_)) {
            *v = ItemValue::Real(v.integer() as f64);
        }
        let fits = match d.kind {
            ItemKind::Bool => matches!(v, ItemValue::Bool(_)),
            ItemKind::Int => matches!(v, ItemValue::Int(_)),
            ItemKind::Real => matches!(v, ItemValue::Real(_)),
            ItemKind::Text | ItemKind::Choice | ItemKind::Reference | ItemKind::Origin => {
                matches!(v, ItemValue::Text(_))
            }
            ItemKind::Array | ItemKind::Terms => true, // refused below: an array item takes an array
        };
        if !fits {
            let want = match d.kind {
                ItemKind::Int | ItemKind::Real => number_or_choice(d),
                ItemKind::Origin => format!("one of {}, {} or a point (x, y, z)", d.choices[0], d.choices[1]),
                k => kind_name(k).to_string(),
            };
            return Err(Error::type_(format!("{} must be {want}", d.name)));
        }
    }
    if d.kind == ItemKind::Choice || (d.kind == ItemKind::Origin && !v.is_array()) || v.is_name() {
        if v.is_array() {
            return Err(Error::type_(format!("{} must be a string", d.name)));
        }
        if !d.choices.iter().any(|c| v.text() == *c) {
            return Err(Error::value(format!(
                "{} must be one of {}, got {}",
                d.name,
                d.choices.join(", "),
                py_str_repr(v.text())
            )));
        }
        return Ok(());
    }
    let array_kind = d.kind == ItemKind::Array || d.kind == ItemKind::Terms || d.kind == ItemKind::Origin;
    if array_kind != v.is_array() {
        return Err(Error::type_(format!("{} must be {}", d.name, kind_name(d.kind))));
    }
    if v.is_array() {
        let mut n: usize = 1;
        for &x in v.shape() {
            if x < 0 {
                return Err(Error::value(format!("{}: a negative extent {x}", d.name)));
            }
            if x > 0 && n > MAX_ARRAY_ELEMENTS / x as usize {
                return Err(Error::value(format!("{}: more than 2^40 elements", d.name)));
            }
            n *= x as usize;
        }
        let have = match v.dtype() {
            DType::Int => v.ints().len(),
            DType::Text => v.texts().len(),
            DType::Real => v.reals().len(),
        };
        if have != n {
            return Err(Error::argument(format!("{}: the data do not fill the shape", d.name)));
        }
        if n == 0 && v.shape().len() == 1 {
            if let Some(shape) = v.shape_mut() {
                *shape = empty_shape(d);
            }
        }
        let alts = split(d.shape, '|');
        if !alts.iter().any(|a| matches(v.shape(), a)) {
            return Err(Error::value(format!(
                "{} must have shape {}, got {}",
                d.name,
                shape_text(d),
                shape_of(v.shape())
            )));
        }
        let want = if d.kind == ItemKind::Array { d.dtype } else { DType::Real };
        if v.dtype() != want {
            if want == DType::Real && v.dtype() == DType::Int {
                let ItemValue::Ints { shape, data } = std::mem::take(v) else { unreachable!("an array of integers") };
                *v = ItemValue::Reals { shape, data: data.iter().map(|&k| k as f64).collect() };
            } else {
                return Err(Error::type_(format!(
                    "{} must be an array of {}",
                    d.name,
                    match want {
                        DType::Int => "integers",
                        DType::Text => "strings",
                        DType::Real => "real numbers",
                    }
                )));
            }
        }
    }
    check_rule(d, v)?;
    if d.rule == ItemRule::Ranges {
        check_ranges(d, v)?;
    }
    Ok(())
}

// element ranges: (n, 2) integers, one-based, first <= last, without overlaps; sorted by their
// first element (unset: every element of the mesh)
fn check_ranges(d: &ItemDef, v: &mut ItemValue) -> Result<()> {
    if v.shape()[0] == 0 {
        return Err(Error::value(format!("{}: the ranges select no elements (None for every element)", d.name)));
    }
    let mut r: Vec<(i64, i64)> = v.ints().chunks_exact(2).map(|c| (c[0], c[1])).collect();
    for &(a, b) in &r {
        if a < 1 || b < a || b > i32::MAX as i64 {
            return Err(Error::value(format!(
                "{}: invalid element range ({a}, {b}) (one-based, first <= last)",
                d.name
            )));
        }
    }
    r.sort();
    for k in 1..r.len() {
        if r[k].0 <= r[k - 1].1 {
            return Err(Error::value(format!(
                "{}: overlapping element ranges ({}, {}) and ({}, {})",
                d.name,
                r[k - 1].0,
                r[k - 1].1,
                r[k].0,
                r[k].1
            )));
        }
    }
    if let ItemValue::Ints { data, .. } = v {
        *data = r.iter().flat_map(|&(a, b)| [a, b]).collect();
    }
    Ok(())
}

// ------------------------------------------------------------------ the surfaces

/// A surface of flat items: the switch item of its method (a body's "interior_surface" and
/// "control_surface", second_order's "free_surface_method") and the prefix of its items
/// ("interior_surface_", "control_surface_", "free_surface_").
#[derive(Clone, Copy, Debug)]
pub struct SurfaceItems {
    pub choice: &'static str,
    pub prefix: &'static str,
}

pub const INTERIOR_SURFACE: SurfaceItems = SurfaceItems { choice: "interior_surface", prefix: "interior_surface_" };
pub const CONTROL_SURFACE: SurfaceItems = SurfaceItems { choice: "control_surface", prefix: "control_surface_" };
pub const FREE_SURFACE: SurfaceItems = SurfaceItems { choice: "free_surface_method", prefix: "free_surface_" };

/// Whether the switches of an object choose an item (an item without a switch always): one of
/// its `switch_values` is the value of its switch (with "v/w" also the second switch's).
pub fn chosen(o: Obj, d: &ItemDef) -> bool {
    if d.switch.is_empty() {
        return true;
    }
    let values: Vec<String> = d.switch.split_whitespace().map(|s| o.get(s).text().to_string()).collect();
    d.switch_values.split_whitespace().any(|t| t.split('/').zip(&values).all(|(want, v)| want == v))
}

// the items that may stay unset where their switch chooses them: the ranges of a mesh (every
// element) and the partitions of the automatic control surface (none)
fn may_be_unset(name: &str) -> bool {
    name.ends_with("ranges") || name.ends_with("partitions")
}

/// The check of the switched items of an object before the problem is built (inertia_input, the
/// methods of the surfaces): the items the switches do not choose are kept unused and not checked;
/// every item they choose that has no default is given (as a body's mesh); ErrorKind::Value naming
/// it and the switch values that choose it.
pub fn check_switched(o: Obj) -> Result<()> {
    for d in &o.type_().items {
        if d.switch.is_empty() || d.fallback.is_some() || may_be_unset(d.name) || !chosen(o, d) || o.is_set(d.name) {
            continue;
        }
        let by: Vec<String> = d.switch.split_whitespace().map(|s| format!("{s} = '{}'", o.get(s).text())).collect();
        // only the switches the value that chooses it names ("cylinder": the first alone)
        let parts = d
            .switch_values
            .split_whitespace()
            .find(|t| t.split('/').zip(d.switch.split_whitespace()).all(|(want, s)| want == o.get(s).text()))
            .map_or(1, |t| t.split('/').count());
        return Err(Error::value(format!("{}{} needs {}", o.prefix(), by[..parts].join(" with "), d.name)));
    }
    Ok(())
}

// ------------------------------------------------------------------ defaults

// the default of an item, read from its schema literal once (the items are 'static)
fn fallback_value(d: &'static ItemDef) -> &'static ItemValue {
    static CACHE: LazyLock<Mutex<HashMap<usize, &'static ItemValue>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let key = d as *const ItemDef as usize;
    if let Some(v) = cache.get(&key) {
        return v;
    }
    let mut v = ItemValue::default();
    if let Some(f) = d.fallback {
        // a schema default that does not read is a bug of the schema (every_default_is_valid)
        let read = || -> Result<ItemValue> {
            let mut at = 0usize;
            let mut v = from_literal(d, &read_literal(f, &mut at)?)?;
            check_value(d, &mut v)?;
            Ok(v)
        };
        v = read().unwrap_or_else(|e| panic!("{e}"));
    }
    let v: &'static ItemValue = Box::leak(Box::new(v));
    cache.insert(key, v);
    v
}

// ------------------------------------------------------------------ objects

/// The handle of an object of a model: the slot of the object in its document and the
/// generation of that slot. The id of an object taken out of the model is refused from then on,
/// also when its slot holds another object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId {
    index: u32,
    generation: u32,
}

impl ObjectId {
    /// The id as one number (the generation in the high 32 bits, the slot in the low ones), for a
    /// client that keeps ids outside Rust (the C ABI).
    pub fn to_bits(self) -> u64 {
        (u64::from(self.generation) << 32) | u64::from(self.index)
    }
    /// The id of `to_bits`; a number that names no object of a model is refused by the model.
    pub fn from_bits(bits: u64) -> ObjectId {
        ObjectId { index: bits as u32, generation: (bits >> 32) as u32 }
    }
}

/// One object of the arena: its type, its place (parent, list, name), the items set and the
/// lists of children.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) type_: &'static TypeDef,
    pub(crate) parent: Option<ObjectId>,
    pub(crate) name: String,
    pub(crate) list: String,
    pub(crate) items: Vec<(String, ItemValue)>, // the items set
    pub(crate) lists: Vec<(String, Vec<ObjectId>)>,
    // the zero-based place of the object in its list in the model this copy was made from (the
    // copy of the enabled objects: an object without a name keeps the position its path names)
    pub(crate) place: Option<u32>,
}

#[derive(Clone, Debug)]
struct Slot {
    generation: u32,
    node: Option<Node>,
}

// the items and children of an object, detached from the arena (copy_items)
struct Tree {
    type_: &'static TypeDef,
    name: String,
    list: String,
    items: Vec<(String, ItemValue)>,
    lists: Vec<(String, Vec<Tree>)>,
}

// the object of a named list in the messages, as the model rules name it
fn singular(list: &str) -> &str {
    match list {
        "bodies" => "body",
        "lids" => "lid",
        "meshes" => "mesh",
        "generalized_modes" => "generalized mode",
        "pressure_modes" => "pressure mode",
        "fixed_dampers" => "fixed damper",
        "dipoles" => "dipole",
        "dipole_dampers" => "dipole damper",
        "tanks" => "tank",
        "pressure_surfaces" => "pressure surface",
        "free_surface_plates" => "free-surface plate",
        _ => list,
    }
}

// the first object outside `skip` and outside the list `dropping` (an object and one of its lists,
// which a clear drops) with a reference that `hit` takes (its target and one of its values)
fn referrer<'a>(
    o: Obj<'a>,
    hit: &dyn Fn(&str, &str) -> bool,
    skip: Option<ObjectId>,
    dropping: Option<(ObjectId, &str)>,
) -> Option<(Obj<'a>, String, String)> {
    if skip == Some(o.id) {
        return None;
    }
    for d in &o.type_().items {
        if is_reference_item(d) && o.is_set(d.name) {
            let v = o.get(d.name);
            if hit(d.target, v.text()) {
                return Some((o, d.target.to_string(), v.text().to_string()));
            }
        }
    }
    for l in &o.type_().lists {
        if dropping == Some((o.id, l.name)) {
            continue;
        }
        for c in o.children(l.name) {
            if let Some(r) = referrer(c, hit, skip, dropping) {
                return Some(r);
            }
        }
    }
    None
}

// every item with a reference to the object at the address (p, a) or to an object within it, in the
// order of the tree (the items of an object, then its children)
fn references(o: Obj, p: &str, a: &str, out: &mut Vec<(ObjectId, &'static str)>) {
    for d in &o.type_().items {
        if is_reference_item(d) && o.is_set(d.name) {
            let v = o.get(d.name);
            if names_within(d.target, v.text(), p, a) {
                out.push((o.id, d.name));
            }
        }
    }
    for l in &o.type_().lists {
        for c in o.children(l.name) {
            references(c, p, a, out);
        }
    }
}

// a reference value with the name `a` of an object renamed to `b`
fn renamed(t: &str, v: &str, p: &str, a: &str, b: &str) -> String {
    if names_within(t, v, p, a) { b.to_string() } else { v.to_string() }
}

/// The value of an origin item (`Obj::origin`): one of its names, or a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum OriginItem<'a> {
    Named(&'a str),
    Point([f64; 3]),
}

/// A borrowed view of one object of a document, for reading.
#[derive(Clone, Copy)]
pub struct Obj<'a> {
    doc: &'a Document,
    id: ObjectId,
}

impl<'a> Obj<'a> {
    fn n(&self) -> &'a Node {
        self.doc.node(self.id)
    }
    pub fn id(&self) -> ObjectId {
        self.id
    }
    pub fn document(&self) -> &'a Document {
        self.doc
    }
    pub fn type_(&self) -> &'static TypeDef {
        self.n().type_
    }
    pub fn name(&self) -> &'a str {
        &self.n().name
    }
    /// The list this object is in ("" for the root).
    pub fn list_name(&self) -> &'a str {
        &self.n().list
    }
    /// The parent (None for the root).
    pub fn parent(&self) -> Option<Obj<'a>> {
        self.n().parent.map(|p| self.doc.obj(p))
    }
    /// The zero-based place of this object in its list (0 for the root).
    pub fn index(&self) -> usize {
        match self.parent() {
            Some(p) => p.children_ids(self.list_name()).iter().position(|&c| c == self.id).unwrap_or(0),
            None => 0,
        }
    }
    /// The zero-based position of this object in its list as its path names it: its index, or in
    /// the copy of the enabled objects of a model its index in that model.
    pub fn position(&self) -> usize {
        match self.n().place {
            Some(p) => p as usize,
            None => self.index(),
        }
    }
    /// The path from the root: "bodies/hull/tanks/2" ("" for the root).
    pub fn path(&self) -> String {
        let Some(parent) = self.parent() else {
            return String::new();
        };
        let up = parent.path();
        let list = self.list_name();
        let l = parent.type_().list(list).expect("the list of a child is in its parent's type");
        let mut seg = list.to_string();
        if !l.single {
            if l.named || !self.name().is_empty() {
                seg += "/";
                seg += self.name();
            } else if parent.children_ids(list).contains(&self.id) {
                seg += &format!("/{}", self.position() + 1);
            }
        }
        if up.is_empty() { seg } else { format!("{up}/{seg}") }
    }
    /// "bodies/hull: " (the root: "") for the messages.
    pub fn prefix(&self) -> String {
        let p = self.path();
        if p.is_empty() { String::new() } else { p + ": " }
    }

    // the items
    pub fn item_def(&self, item: &str) -> Result<&'static ItemDef> {
        let t = self.type_();
        let Some(d) = t.item(item) else {
            return Err(Error::argument(format!("{}no item '{}' in a {}", self.prefix(), item, t.name)));
        };
        if !self.applies(d) {
            return Err(Error::argument(format!(
                "{}a '{}' mesh has no item '{}'",
                self.prefix(),
                self.get("kind").text(),
                item
            )));
        }
        Ok(d)
    }
    /// The text of an item value given as bytes (a C string), checked as the model checks a text
    /// when the item is set: ErrorKind::Value "<prefix><item>: the text is not valid UTF-8" (the
    /// name item: "<prefix>the text is not valid UTF-8"), or the NUL check. Fails with item_def's
    /// errors first (an unknown item).
    pub fn item_text(&self, item: &str, bytes: &[u8]) -> Result<String> {
        let d = self.item_def(item)?;
        let at = if d.rule == ItemRule::Name { self.prefix() } else { format!("{}{}: ", self.prefix(), d.name) };
        Ok(check_utf8(&at, bytes)?.to_string())
    }
    /// The value, the default where it is not set (an unset optional item: set = false).
    /// The value of an item of this object's type (an item the type does not have is a bug: the
    /// entry points check the names they are given with `item_def`).
    pub fn get(&self, item: &str) -> ItemValue {
        let t = self.type_();
        let Some(d) = t.item(item) else {
            panic!("{}no item '{}' in a {}", self.prefix(), item, t.name);
        };
        if d.rule == ItemRule::Name {
            return ItemValue::Text(self.name().to_string());
        }
        for (k, v) in &self.n().items {
            if k == item {
                return v.clone();
            }
        }
        fallback_value(d).clone()
    }
    pub fn is_set(&self, item: &str) -> bool {
        self.n().items.iter().any(|(k, _)| k == item)
    }
    // the typed reads of the build (document_build.rs, document_enabled.rs): the value of an item
    // of this object's type, its default where it is not set, checked against the item's kind. An
    // item the type does not have or of another kind is a bug of the caller (a panic, as `get`); an
    // item without a value (unset, no default) is ErrorKind::Internal naming it, never a zero: the
    // rules give every item a value before the build reads it. An item that may have none is asked
    // first (`has`), or read as `Option` (`optional_reals`, `optional_ints`, `mesh`, `body`).

    // the value of an item (set, or its default) of one of the kinds, None when it has none
    fn value_of(&self, item: &str, kinds: &[ItemKind]) -> Option<&'a ItemValue> {
        let t = self.type_();
        let Some(d) = t.item(item) else {
            panic!("{}no item '{}' in a {}", self.prefix(), item, t.name);
        };
        assert!(
            kinds.contains(&d.kind) && d.rule != ItemRule::Name,
            "{}{} is an item of kind {}, not {}",
            self.prefix(),
            item,
            d.kind.name(),
            kinds.iter().map(|k| k.name()).collect::<Vec<_>>().join(" or ")
        );
        let v = match self.n().items.iter().find(|(k, _)| k == item) {
            Some((_, v)) => v,
            None => fallback_value(d),
        };
        v.is_set().then_some(v)
    }
    // the value of an item that must have one: ErrorKind::Internal otherwise
    fn required(&self, item: &str, kinds: &[ItemKind]) -> Result<&'a ItemValue> {
        self.value_of(item, kinds).ok_or_else(|| self.no_value(item))
    }
    fn no_value(&self, item: &str) -> Error {
        Error::internal(format!("{}{item} has no value (unset, without a default)", self.prefix()))
    }
    // a number given as one of its names (an Int or Real item with choices) where a number is read
    fn number(&self, item: &str, v: &ItemValue) -> Result<()> {
        if v.is_name() {
            return Err(Error::internal(format!("{}{item} is {}, not a number", self.prefix(), py_str_repr(v.text()))));
        }
        Ok(())
    }
    /// Whether an item has a value: it is set, or it has a default.
    pub(crate) fn has(&self, item: &str) -> bool {
        let t = self.type_();
        let Some(d) = t.item(item) else {
            panic!("{}no item '{}' in a {}", self.prefix(), item, t.name);
        };
        d.rule == ItemRule::Name || self.is_set(item) || fallback_value(d).is_set()
    }
    /// A bool item.
    pub(crate) fn boolean(&self, item: &str) -> Result<bool> {
        Ok(self.required(item, &[ItemKind::Bool])?.boolean())
    }
    /// An int item given as a number.
    pub(crate) fn integer(&self, item: &str) -> Result<i64> {
        let v = self.required(item, &[ItemKind::Int])?;
        self.number(item, v)?;
        Ok(v.integer())
    }
    /// An int item that may be given by one of its names: None for a name ("auto").
    pub(crate) fn integer_or_name(&self, item: &str) -> Result<Option<i64>> {
        let v = self.required(item, &[ItemKind::Int])?;
        Ok((!v.is_name()).then_some(v.integer()))
    }
    /// A real item given as a number.
    pub(crate) fn real(&self, item: &str) -> Result<f64> {
        let v = self.required(item, &[ItemKind::Real])?;
        self.number(item, v)?;
        Ok(v.real())
    }
    /// A real item that may be given by one of its names: None for a name ("auto", "walls_top",
    /// "displaced").
    pub(crate) fn real_or_name(&self, item: &str) -> Result<Option<f64>> {
        let v = self.required(item, &[ItemKind::Real])?;
        Ok((!v.is_name()).then_some(v.real()))
    }
    /// A choice item: the name chosen.
    pub(crate) fn choice(&self, item: &str) -> Result<&'a str> {
        Ok(self.required(item, &[ItemKind::Choice])?.text())
    }
    /// An origin item: one of its names, or a point.
    pub(crate) fn origin(&self, item: &str) -> Result<OriginItem<'a>> {
        let v = self.required(item, &[ItemKind::Origin])?;
        Ok(match v {
            ItemValue::Reals { data, .. } => OriginItem::Point([data[0], data[1], data[2]]),
            _ => OriginItem::Named(v.text()),
        })
    }
    /// The elements of an array of reals or of a terms item (C order).
    pub(crate) fn reals(&self, item: &str) -> Result<&'a [f64]> {
        self.optional_reals(item).ok_or_else(|| self.no_value(item))
    }
    /// The elements of an array of reals or of a terms item (C order), None when it has no value
    /// (an optional item: unset, without a default).
    pub(crate) fn optional_reals(&self, item: &str) -> Option<&'a [f64]> {
        let v = self.value_of(item, &[ItemKind::Array, ItemKind::Terms])?;
        let ItemValue::Reals { data, .. } = v else { panic!("{}{item} is not an array of reals", self.prefix()) };
        Some(data)
    }
    /// The elements of an array of integers (C order).
    pub(crate) fn ints(&self, item: &str) -> Result<&'a [i64]> {
        self.optional_ints(item).ok_or_else(|| self.no_value(item))
    }
    /// The elements of an array of integers (C order), None when it has no value (an optional
    /// item: unset, without a default).
    pub(crate) fn optional_ints(&self, item: &str) -> Option<&'a [i64]> {
        let v = self.value_of(item, &[ItemKind::Array])?;
        let ItemValue::Ints { data, .. } = v else { panic!("{}{item} is not an array of integers", self.prefix()) };
        Some(data)
    }
    /// The elements of an array of texts.
    pub(crate) fn texts(&self, item: &str) -> Result<&'a [String]> {
        let v = self.required(item, &[ItemKind::Array])?;
        let ItemValue::Texts { data, .. } = v else { panic!("{}{item} is not an array of texts", self.prefix()) };
        Ok(data)
    }
    /// The shape of an array or terms item.
    pub(crate) fn shape(&self, item: &str) -> Result<&'a [i64]> {
        Ok(self.required(item, &[ItemKind::Array, ItemKind::Terms])?.shape())
    }
    /// A reference to a mesh: its name, None when it is unset.
    pub(crate) fn mesh(&self, item: &str) -> Option<&'a str> {
        self.reference(item, "meshes")
    }
    /// A reference to a body: its name, None when it is unset.
    pub(crate) fn body(&self, item: &str) -> Option<&'a str> {
        self.reference(item, "bodies")
    }
    fn reference(&self, item: &str, target: &str) -> Option<&'a str> {
        let d = self.type_().item(item).filter(|d| d.kind == ItemKind::Reference);
        if let Some(d) = d {
            assert!(d.target == target, "{}{item} names an object of {}, not of {target}", self.prefix(), d.target);
        }
        let v = self.value_of(item, &[ItemKind::Reference])?;
        Some(v.text())
    }
    /// The kind of a mesh object (the schema checks the choice).
    pub fn mesh_kind(&self) -> MeshKind {
        MeshKind::of(self.get("kind").text()).expect("the schema checks the kind of a mesh")
    }
    /// Whether the item applies (a mesh item of another kind does not).
    pub fn applies(&self, d: &ItemDef) -> bool {
        if d.kinds.is_empty() {
            return true;
        }
        let kind = format!(" {} ", self.get("kind").text());
        format!(" {} ", d.kinds).contains(&kind)
    }

    // the children
    /// The ids of the children of a list of this object's type (a list the type does not have is a
    /// bug; a list name from outside goes through `try_children`).
    pub fn children_ids(&self, list: &str) -> &'a [ObjectId] {
        match self.try_children(list) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        }
    }
    /// The children of a list of this object's type, in order.
    pub fn children(&self, list: &str) -> Vec<Obj<'a>> {
        self.children_ids(list).iter().map(|&c| self.doc.obj(c)).collect()
    }
    pub fn try_children(&self, list: &str) -> Result<&'a [ObjectId]> {
        for (k, v) in &self.n().lists {
            if k == list {
                return Ok(v);
            }
        }
        Err(Error::argument(format!("{}no list '{}' in a {}", self.prefix(), list, self.type_().name)))
    }
    /// The child of that name (the child of a single list whatever its name).
    pub fn child(&self, list: &str, name: &str) -> Option<Obj<'a>> {
        let l = self.type_().list(list)?;
        self.children(list).into_iter().find(|c| l.single || c.name() == name)
    }
}

/// A view of one object of a document for changing it (the changes of Document on its id).
pub struct ObjMut<'a> {
    doc: &'a mut Document,
    id: ObjectId,
}

impl ObjMut<'_> {
    pub fn id(&self) -> ObjectId {
        self.id
    }
    pub fn as_obj(&self) -> Obj<'_> {
        self.doc.obj(self.id)
    }
    pub(crate) fn doc_mut(&mut self) -> &mut Document {
        self.doc
    }
    pub fn get(&self, item: &str) -> ItemValue {
        self.as_obj().get(item)
    }
    pub fn set(&mut self, item: &str, v: ItemValue) -> Result<()> {
        self.doc.set(self.id, item, v)
    }
    /// A new child (index -1: at the end), as a view for changing it.
    pub fn add(&mut self, list: &str, type_: &str, name: &str, index: i64) -> Result<ObjMut<'_>> {
        let c = self.doc.add_child(self.id, list, type_, name, index)?;
        Ok(ObjMut { doc: &mut *self.doc, id: c })
    }
}

// ------------------------------------------------------------------ the document

/// The model: the objects (the root, the model, first) and its binary record.
#[derive(Clone, Debug)]
pub struct Document {
    slots: Vec<Slot>,
    free: Vec<u32>,
    root: ObjectId,
    revision: u64,
}

impl Default for Document {
    fn default() -> Self {
        Document::new()
    }
}

impl Document {
    /// A new model: the root and the children its lists always have.
    pub fn new() -> Document {
        let mut d =
            Document { slots: Vec::new(), free: Vec::new(), root: ObjectId { index: 0, generation: 0 }, revision: 0 };
        d.root = d.make(schema_type("Model"), None, String::new(), String::new());
        d
    }

    // ---- the arena

    fn alloc(&mut self, node: Node) -> ObjectId {
        if let Some(i) = self.free.pop() {
            let s = &mut self.slots[i as usize];
            s.node = Some(node);
            return ObjectId { index: i, generation: s.generation };
        }
        self.slots.push(Slot { generation: 0, node: Some(node) });
        ObjectId { index: (self.slots.len() - 1) as u32, generation: 0 }
    }
    // a new object with the children its lists always hold (the revision unchanged)
    fn make(&mut self, type_: &'static TypeDef, parent: Option<ObjectId>, name: String, list: String) -> ObjectId {
        let id = self.alloc(Node { type_, parent, name, list, items: Vec::new(), lists: Vec::new(), place: None });
        let mut lists = Vec::new();
        for l in &type_.lists {
            let mut v = Vec::new();
            if l.always {
                v.push(self.make(schema_type(l.types[0]), Some(id), String::new(), l.name.to_string()));
            }
            lists.push((l.name.to_string(), v));
        }
        self.node_mut(id).lists = lists;
        id
    }
    /// Takes an object out of the model: it and its descendants leave the arena, and their ids are
    /// refused from then on.
    fn bury(&mut self, id: ObjectId) {
        let Some(s) = self.slots.get_mut(id.index as usize) else { return };
        if s.generation != id.generation {
            return;
        }
        let Some(node) = s.node.take() else { return };
        s.generation = s.generation.wrapping_add(1);
        self.free.push(id.index);
        for (_, v) in node.lists {
            for c in v {
                self.bury(c);
            }
        }
    }
    /// Whether id names an object of this model (not one taken out of it).
    pub fn contains(&self, id: ObjectId) -> bool {
        self.slots.get(id.index as usize).is_some_and(|s| s.generation == id.generation && s.node.is_some())
    }
    pub(crate) fn node(&self, id: ObjectId) -> &Node {
        let s = self.slots.get(id.index as usize).filter(|s| s.generation == id.generation);
        s.and_then(|s| s.node.as_ref()).expect("an object of this model")
    }
    pub(crate) fn node_mut(&mut self, id: ObjectId) -> &mut Node {
        let s = self.slots.get_mut(id.index as usize).filter(|s| s.generation == id.generation);
        s.and_then(|s| s.node.as_mut()).expect("an object of this model")
    }
    // the children of a list of an object of this model (a list its type does not have is a bug)
    fn list_mut(&mut self, id: ObjectId, list: &str) -> &mut Vec<ObjectId> {
        if let Some(i) = self.node(id).lists.iter().position(|(k, _)| k == list) {
            return &mut self.node_mut(id).lists[i].1;
        }
        let o = self.obj(id);
        panic!("{}no list '{}' in a {}", o.prefix(), list, o.type_().name);
    }

    // ---- views

    /// The view of an object of this model (an id of another model, or of an object taken out of
    /// it, is a bug: ids from outside are checked with `contains`, or read with `get_obj`).
    pub(crate) fn obj(&self, id: ObjectId) -> Obj<'_> {
        let _ = self.node(id);
        Obj { doc: self, id }
    }
    /// The view of an object, None when id is not one of this model.
    pub fn get_obj(&self, id: ObjectId) -> Option<Obj<'_>> {
        if self.contains(id) { Some(Obj { doc: self, id }) } else { None }
    }
    pub(crate) fn obj_mut(&mut self, id: ObjectId) -> ObjMut<'_> {
        let _ = self.node(id);
        ObjMut { doc: self, id }
    }
    pub fn root(&self) -> Obj<'_> {
        self.obj(self.root)
    }
    pub fn root_id(&self) -> ObjectId {
        self.root
    }
    // the view of an object given to a change of the document: ErrorKind::Argument for an id of
    // another document or of an object taken out of this one
    fn need(&self, id: ObjectId) -> Result<Obj<'_>> {
        self.get_obj(id).ok_or_else(|| Error::argument("the object is not one of this model (removed, or of another)"))
    }

    // ---- the items

    /// Checks the value alone, then sets it; a failure leaves the item unchanged.
    pub fn set(&mut self, id: ObjectId, item: &str, mut v: ItemValue) -> Result<()> {
        let me = self.need(id)?;
        let d = me.item_def(item)?;
        if d.rule == ItemRule::Name {
            if !v.is_set() || v.is_array() {
                return Err(Error::type_(format!("{}name must be a string", me.prefix())));
            }
            return self.rename(id, v.text());
        }
        if !v.is_set() {
            return self.reset(id, item);
        }
        let checked = (|| -> Result<()> {
            check_value(d, &mut v)?;
            let t = me.type_();
            if is_reference_item(d) {
                if let Err(m) = resolve_reference(me.document().root(), d.target, v.text()) {
                    return Err(Error::value(format!("{}: {m}", d.name)));
                }
            }
            // a low-order mesh: its panels checked with the values it will have, whichever of them
            // changes, so that no mesh is kept that could not be given (and read back) as it is
            if t.name == "Mesh"
                && (item == "vertices" || item == "kind" || item == "symmetry" || item == "reference_length")
            {
                let now = |n: &str| -> ItemValue {
                    if item == n {
                        return v.clone();
                    }
                    match t.item(n) {
                        Some(x) if me.applies(x) => me.get(n),
                        _ => ItemValue::default(),
                    }
                };
                let kind = now("kind");
                let verts = now("vertices");
                if kind.text() == MeshKind::Panels.name()
                    && verts.is_set()
                    && verts.is_array()
                    && !verts.shape().is_empty()
                    && verts.shape()[0] > 0
                {
                    let length = now("reference_length");
                    let sym = now("symmetry");
                    let mut m = LowOrderMeshDesc {
                        panel_count: verts.shape()[0] as i32,
                        reference_length: if length.is_set() { length.real() } else { 1.0 },
                        gravity: STANDARD_GRAVITY,
                        ..LowOrderMeshDesc::default()
                    };
                    m.vertices = verts.reals().to_vec();
                    let symmetry = t.item("symmetry").expect("a Mesh has symmetry");
                    let s = symmetry_flags(if sym.is_set() { &sym } else { fallback_value(symmetry) });
                    m.symmetry_x = s[0];
                    m.symmetry_y = s[1];
                    build_mesh(&m)?;
                }
            }
            Ok(())
        })();
        if let Err(e) = checked {
            let message = format!("{}{}", me.prefix(), e.message());
            return Err(e.with_message(message));
        }
        let t = me.type_();
        if t.name == "Mesh" && item == "kind" {
            let kind = v.text().to_string();
            self.node_mut(id).items.retain(|(n, _)| {
                let x = t.item(n).expect("a set item is in the schema");
                !(!x.kinds.is_empty() && !format!(" {} ", x.kinds).contains(&format!(" {kind} ")))
            });
            let keep = MeshKind::of(&kind).and_then(MeshKind::patch_type).unwrap_or("");
            let patches = std::mem::take(self.list_mut(id, "patches"));
            let (kept, other): (Vec<ObjectId>, Vec<ObjectId>) =
                patches.into_iter().partition(|&p| self.node(p).type_.name == keep);
            *self.list_mut(id, "patches") = kept;
            for o in other {
                self.bury(o);
            }
        }
        let node = self.node_mut(id);
        match node.items.iter_mut().find(|(k, _)| k == item) {
            Some(x) => x.1 = v,
            None => node.items.push((item.to_string(), v)),
        }
        self.changed();
        Ok(())
    }
    /// Returns an item to its default (or unsets it).
    pub fn reset(&mut self, id: ObjectId, item: &str) -> Result<()> {
        let o = self.need(id)?;
        let d = o.item_def(item)?;
        if d.rule == ItemRule::Name {
            return Err(Error::argument(format!("{}the name cannot be reset", o.prefix())));
        }
        self.node_mut(id).items.retain(|(k, _)| k != item);
        self.changed();
        Ok(())
    }

    // ---- the children

    /// A new child of object id (index -1: at the end).
    pub fn add_child(&mut self, id: ObjectId, list: &str, type_: &str, name: &str, index: i64) -> Result<ObjectId> {
        let o = self.need(id)?;
        let ot = o.type_();
        let Some(l) = ot.list(list) else {
            return Err(Error::argument(format!("{}no list '{}' in a {}", o.prefix(), list, ot.name)));
        };
        let size = o.children_ids(list).len();
        let t: &str = if type_.is_empty() { l.types[0] } else { type_ };
        if !l.types.contains(&t) {
            return Err(Error::argument(format!("{}{} holds no {}", o.prefix(), list, t)));
        }
        if ot.name == "Mesh" {
            let kind = o.mesh_kind();
            if kind.patch_type() != Some(t) {
                return Err(Error::argument(format!("{}a '{}' mesh has no {}", o.prefix(), kind.name(), t)));
            }
        }
        if l.single && size != 0 {
            return Err(Error::value(format!("{}{} is already given", o.prefix(), list)));
        }
        if !l.named && !l.optional_names && !name.is_empty() {
            return Err(Error::argument(format!("{}the objects of {} have no names", o.prefix(), list)));
        }
        if l.named || !name.is_empty() {
            check_name(&format!("{}{}: ", o.prefix(), list), name)?;
            check_optional_name(&format!("{}{}: ", o.prefix(), list), l, name)?;
            if o.child(list, name).is_some() {
                return Err(Error::value(format!(
                    "{}duplicate {} name {}",
                    o.prefix(),
                    singular(list),
                    py_str_repr(name)
                )));
            }
        }
        if index < -1 || index > size as i64 {
            return Err(Error::argument(format!("{}{}: index out of range", o.prefix(), list)));
        }
        let c = self.make(schema_type(t), Some(id), name.to_string(), list.to_string());
        let at = if index < 0 { size } else { index as usize };
        self.list_mut(id, list).insert(at, c);
        self.changed();
        Ok(c)
    }
    // the checks of a change of child c of object parent: a child of it
    fn need_child(&self, parent: ObjectId, c: ObjectId) -> Result<()> {
        if !self.contains(c) || self.node(c).parent != Some(parent) {
            return Err(Error::argument("not a child of this object"));
        }
        Ok(())
    }
    /// Removes child c of object parent with its children.
    pub fn remove_child(&mut self, parent: ObjectId, c: ObjectId) -> Result<()> {
        self.need_child(parent, c)?;
        let p = self.obj(parent);
        let cr = self.obj(c);
        let l = p.type_().list(cr.list_name()).expect("the list of a child is in its parent's type");
        if l.always {
            return Err(Error::argument(format!("{}{} cannot be removed", p.prefix(), cr.list_name())));
        }
        // an object another one refers to stays: a mesh of a body, the body of a lid or of a coupling
        // (also through a reference to an object within it)
        if let Some((ap, aa)) = address(cr) {
            if let Some((r, _, _)) = referrer(self.root(), &|t, x| names_within(t, x, &ap, &aa), Some(c), None) {
                return Err(Error::value(format!(
                    "{}is used by {}: change or remove that reference first",
                    cr.prefix(),
                    r.path()
                )));
            }
        }
        let list = cr.list_name().to_string();
        let v = self.list_mut(parent, &list);
        let Some(i) = v.iter().position(|&p| p == c) else {
            return Err(Error::argument("not a child of this object"));
        };
        v.remove(i);
        self.bury(c);
        self.changed();
        Ok(())
    }
    /// Removes an object with its children.
    pub fn remove(&mut self, id: ObjectId) -> Result<()> {
        self.need(id)?;
        let Some(p) = self.node(id).parent else {
            return Err(Error::argument("the model itself cannot be removed"));
        };
        self.remove_child(p, id)
    }
    /// Moves child c of object parent to a zero-based position of its list.
    pub fn move_child(&mut self, parent: ObjectId, c: ObjectId, index: i64) -> Result<()> {
        self.need_child(parent, c)?;
        let p = self.obj(parent);
        let list = self.obj(c).list_name().to_string();
        let size = p.children_ids(&list).len();
        if index < 0 || index >= size as i64 {
            return Err(Error::argument(format!("{}{}: index out of range", p.prefix(), list)));
        }
        let v = self.list_mut(parent, &list);
        let Some(i) = v.iter().position(|&p| p == c) else {
            return Err(Error::argument("not a child of this object"));
        };
        let keep = v.remove(i);
        v.insert(index as usize, keep);
        self.changed();
        Ok(())
    }
    /// Moves an object to a zero-based position of its list (`move` is a keyword).
    pub fn move_(&mut self, id: ObjectId, index: i64) -> Result<()> {
        self.need(id)?;
        let Some(p) = self.node(id).parent else {
            return Err(Error::argument("the model itself cannot be moved"));
        };
        self.move_child(p, id, index)
    }
    /// Renames an object of a named list or of a list with optional names ("" removes the name of the
    /// latter); the references to an object of the model's lists follow.
    pub fn rename(&mut self, id: ObjectId, name: &str) -> Result<()> {
        let o = self.need(id)?;
        let Some(parent) = o.parent() else {
            return Err(Error::argument("the model has no name"));
        };
        let list = o.list_name();
        let l = parent.type_().list(list).expect("the list of a child is in its parent's type");
        if !l.named && !l.optional_names {
            return Err(Error::argument(format!("{}the objects of {} have no names", o.prefix(), list)));
        }
        if name == o.name() {
            return Ok(());
        }
        // an object of a list with optional names loses its name with ""
        if l.named || !name.is_empty() {
            check_name(&o.prefix(), name)?;
            check_optional_name(&o.prefix(), l, name)?;
        }
        if !name.is_empty() && parent.child(list, name).is_some() {
            return Err(Error::value(format!(
                "{}duplicate {} name {}",
                parent.prefix(),
                singular(list),
                py_str_repr(name)
            )));
        }
        // the references to it (or to an object within it) follow; one that would lose its name
        // while referred to is refused
        let before = address(o);
        if let Some((p, a)) = &before {
            if name.is_empty() {
                if let Some((r, _, _)) = referrer(self.root(), &|t, x| names_within(t, x, p, a), None, None) {
                    return Err(Error::value(format!(
                        "{}is used by {}: an object referred to keeps its name",
                        o.prefix(),
                        r.path()
                    )));
                }
            }
        }
        let mut refs = Vec::new();
        if let Some((p, a)) = &before {
            references(self.root(), p, a, &mut refs);
        }
        self.node_mut(id).name = name.to_string();
        if let (Some((p, a)), Some((_, b))) = (&before, address(self.obj(id))) {
            for (o, item) in refs {
                let d = self.obj(o).item_def(item)?;
                let v = self.obj(o).get(item);
                let v = ItemValue::Text(renamed(d.target, v.text(), p, a, &b));
                self.set(o, item, v)?;
            }
        }
        self.revision += 1;
        Ok(())
    }

    // ---- the clear of a list

    /// Removes every object of a list, all or nothing: refused (nothing removed) while another
    /// object of the model refers to one of them or to an object within them.
    pub fn clear_list(&mut self, id: ObjectId, list: &str) -> Result<()> {
        let o = self.need(id)?;
        let Some(l) = o.type_().list(list) else {
            return Err(Error::argument(format!("{}no list '{}' in a {}", o.prefix(), list, o.type_().name)));
        };
        if l.always {
            return Err(Error::argument(format!("{}{} always holds its object: it is not emptied", o.prefix(), list)));
        }
        // the list path of the children and the address of their parent ("" the root); none for a
        // parent without an address, whose children no reference can name
        let at = if o.parent().is_none() {
            Some((list.to_string(), String::new()))
        } else {
            address(o).map(|(p, a)| (format!("{p}/{list}"), a))
        };
        if let Some((p, a)) = at {
            // no reference from outside the list into it (or to an object within its objects)
            let into = |t: &str, _: &str| names_into(t, &p, &a);
            if let Some((r, t, x)) = referrer(self.root(), &into, None, Some((id, list))) {
                return Err(Error::value(format!(
                    "{} refers to {}: the {} are not cleared while an object refers to them",
                    r.path(),
                    reference_path(&t, &x),
                    list
                )));
            }
        }
        for c in std::mem::take(self.list_mut(id, list)) {
            self.bury(c);
        }
        self.changed();
        Ok(())
    }

    // ---- copies

    /// The items and children of another object of the same type (its name and place kept).
    pub fn copy_items(&mut self, id: ObjectId, from: ObjectId) -> Result<()> {
        let o = self.need(id)?;
        let t = o.type_();
        if !std::ptr::eq(self.need(from)?.type_(), t) {
            return Err(Error::argument(format!("{}a {} takes the items of a {}", o.prefix(), t.name, t.name)));
        }
        let name = o.name().to_string();
        let list = o.list_name().to_string();
        let tree = self.snapshot(from);
        self.graft(id, tree);
        let node = self.node_mut(id);
        node.name = name;
        node.list = list;
        self.changed();
        Ok(())
    }
    // the items and children of an object
    fn snapshot(&self, id: ObjectId) -> Tree {
        let n = self.node(id);
        Tree {
            type_: n.type_,
            name: n.name.clone(),
            list: n.list.clone(),
            items: n.items.clone(),
            lists: n.lists.iter().map(|(k, v)| (k.clone(), v.iter().map(|&c| self.snapshot(c)).collect())).collect(),
        }
    }
    // object id becomes the copy of t: its name, list and items, its children dropped and those of t
    // made
    fn graft(&mut self, id: ObjectId, t: Tree) {
        {
            let node = self.node_mut(id);
            node.name = t.name;
            node.list = t.list;
            node.items = t.items;
        }
        let mut copies = t.lists;
        for k in 0..self.node(id).lists.len() {
            let old = std::mem::take(&mut self.node_mut(id).lists[k].1);
            for c in old {
                self.bury(c);
            }
            let key = self.node(id).lists[k].0.clone();
            let Some(i) = copies.iter().position(|(n, _)| *n == key) else { continue };
            for c in std::mem::take(&mut copies[i].1) {
                let lists = c.type_.lists.iter().map(|l| (l.name.to_string(), Vec::new())).collect();
                let x = self.alloc(Node {
                    type_: c.type_,
                    parent: Some(id),
                    name: c.name.clone(),
                    list: c.list.clone(),
                    items: Vec::new(),
                    lists,
                    place: None,
                });
                self.graft(x, c);
                self.node_mut(id).lists[k].1.push(x);
            }
        }
    }

    // ---- the model

    /// Every object and item back to a new model (the revision moves on; every id of the old
    /// objects, the root's too, is refused from then on).
    pub fn clear(&mut self) {
        let old = self.root;
        self.root = self.make(schema_type("Model"), None, String::new(), String::new());
        self.bury(old);
        self.changed();
    }
    /// The object of a path ("" the root); None when there is none.
    pub fn find(&self, path: &str) -> Option<ObjectId> {
        let mut o = self.root();
        if path.is_empty() {
            return Some(o.id);
        }
        let seg = split(path, '/');
        let mut k = 0usize;
        while k < seg.len() {
            let l = o.type_().list(&seg[k])?;
            let v = o.children_ids(&seg[k]);
            if l.single {
                o = self.obj(*v.first()?);
                k += 1;
                continue;
            }
            k += 1;
            if k == seg.len() {
                return None;
            }
            if l.named || (l.optional_names && !seg[k].is_empty() && !all_digits(&seg[k])) {
                o = o.child(&seg[k - 1], &seg[k])?;
                k += 1;
                continue;
            }
            if !all_digits(&seg[k]) || seg[k].len() > 10 {
                return None;
            }
            let i = seg[k].parse::<u64>().unwrap_or(0) as usize;
            if i < 1 || i > v.len() {
                return None;
            }
            o = self.obj(v[i - 1]);
            k += 1;
        }
        Some(o.id)
    }
    /// Incremented by every change (the cache of the built problem).
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn changed(&mut self) {
        self.revision += 1;
    }

    /// Every object keeps its present position as the position its path names (`Obj::position`),
    /// for a copy of the model whose objects are then taken out (`take_out`).
    pub(crate) fn keep_places(&mut self) {
        for i in 0..self.slots.len() {
            let Some(node) = self.slots[i].node.as_ref() else { continue };
            let Some(parent) = node.parent else { continue };
            let id = ObjectId { index: i as u32, generation: self.slots[i].generation };
            let list = node.list.clone();
            let p = self
                .node(parent)
                .lists
                .iter()
                .find(|(k, _)| *k == list)
                .and_then(|(_, v)| v.iter().position(|&c| c == id));
            if let (Some(p), Some(node)) = (p, self.slots[i].node.as_mut()) {
                node.place = Some(p as u32);
            }
        }
    }
    /// Takes an object out of the model with its children, whatever refers to it (the copy of the
    /// enabled objects, which then drops or keeps those references itself).
    pub(crate) fn take_out(&mut self, id: ObjectId) {
        let Some(parent) = self.node(id).parent else { return };
        let list = self.node(id).list.clone();
        self.list_mut(parent, &list).retain(|&c| c != id);
        self.bury(id);
        self.changed();
    }

    // record(), load_record(): document_record.rs
}

#[cfg(test)]
mod tests {
    use super::*;

    // every default of the schema reads as a literal and passes the checks of its item
    #[test]
    fn every_default_is_valid() {
        for t in schema() {
            for d in &t.items {
                let v = fallback_value(d);
                assert_eq!(v.is_set(), d.fallback.is_some(), "{}.{}", t.name, d.name);
            }
        }
        let modes = fallback_value(schema_type("Body").item("modes").unwrap());
        assert_eq!(modes.texts().len(), 6);
        assert_eq!(modes.shape(), vec![6]);
        let terms = fallback_value(schema_type("ModePiece").item("u").unwrap());
        assert_eq!(terms.shape(), vec![0, 4]);
        let every = fallback_value(schema_type("ModePiece").item("ranges").unwrap());
        assert!(!every.is_set()); // unset: every element
        let formulation = fallback_value(schema_type("FirstOrder").item("diffraction_formulation").unwrap());
        assert_eq!(formulation.text(), "auto");
        let p = fallback_value(schema_type("AnalyticPatch").item("p").unwrap());
        assert_eq!(p.reals(), vec![0.0; 4]);
    }

    #[test]
    fn literals_and_shapes() -> Result<()> {
        let mut at = 0;
        let l = read_literal("[[1, 2.5], [-3, inf]]", &mut at)?;
        assert!(l.is_array() && l.items.len() == 2);
        assert_eq!(l.items[1].items[0].integer, -3);
        assert!(l.items[1].items[1].number.is_infinite());
        assert!(matches(&[6, 12], "6,6N"));
        assert!(!matches(&[6, 0], "6,6N"));
        assert!(matches(&[4, 4], "M,M"));
        assert!(!matches(&[4, 5], "M,M"));
        assert!(matches(&[2, 4, 3], "P,4,3"));
        assert_eq!(shape_of(&[3]), "(3,)");
        assert_eq!(shape_of(&[0, 4, 3]), "(0, 4, 3)");
        Ok(())
    }

    // the typed reads: the value or its default; an item without a value (unset, no default) an
    // internal error, a name where a number is read too; an item of another kind a bug (a panic)
    #[test]
    fn the_typed_reads_check_the_kind() {
        use crate::error::ErrorKind;
        let mut d = Document::new();
        let root = d.root_id();
        let b = d.add_child(root, "bodies", "Body", "hull", -1).unwrap();
        let o = d.obj(b);
        assert_eq!(o.real("yaw").unwrap(), 0.0);
        assert_eq!(o.reals("position").unwrap(), [0.0; 3]);
        assert_eq!(o.choice("interior_surface").unwrap(), "detect");
        assert_eq!(o.texts("modes").unwrap().len(), 6);
        assert_eq!(o.origin("inertia_origin").unwrap(), OriginItem::Named("center_of_gravity"));
        assert!(o.boolean("enabled").unwrap());
        assert!(!o.has("trim") && o.optional_reals("trim").is_none() && o.mesh("mesh").is_none());
        assert_eq!(o.reals("trim").unwrap_err().kind(), ErrorKind::Internal);
        assert_eq!(o.real_or_name("mass").unwrap_err().kind(), ErrorKind::Internal);
        let second = d.obj(root).child("second_order", "").unwrap();
        assert_eq!(second.real_or_name("free_surface_panel_size").unwrap(), None);
        assert_eq!(second.real("free_surface_panel_size").unwrap_err().kind(), ErrorKind::Internal);
        assert_eq!(second.integer_or_name("quadrature_radial_nodes").unwrap(), Some(12));
        assert_eq!(second.integer("quadrature_azimuthal_nodes").unwrap_err().kind(), ErrorKind::Internal);
        let wrong = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| o.real("position")));
        assert!(wrong.is_err());
        let wrong = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| o.body("mesh")));
        assert!(wrong.is_err());
    }
}
