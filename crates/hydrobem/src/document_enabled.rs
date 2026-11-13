//! The model as it is solved: a copy of the document without the objects whose item `enabled` is
//! false (each with everything under it), as if they were absent, and the numbering of the bodies,
//! lids and field points of the copy in the model. The problem is built from the copy, so that the
//! engine never sees a suppressed object; the locations of its errors and warnings are mapped back
//! to the objects of the model. Each problem built from a model has its copy (`Scope`,
//! `scoped_document`: the model's, a body's queries, a body's mesh, the free surface's queries),
//! made by one walk of the tree (`suppressed` with its `Walk`).
//!
//! What goes with a suppressed object: a coupling that names a suppressed body; the field points
//! of a suppressed tank, or of a tank of a suppressed body (the points kept are the field points of
//! the solve: those of the sea, then those of the tanks kept); the field outputs when no field
//! point is left. A lid placed with a suppressed body is refused: it is placed by that body's
//! position and elements.
//!
//! The field points of the solve are the points of the sea (first_order.field_points), then the
//! points of every tank of the bodies in order; a message or location of one names it in the list
//! of its owner: a point of the sea by its number in first_order.field_points, a point of a tank by
//! its number in the tank's field_points and the tank ("field point 2 of bodies/hull/tanks/aft",
//! the location with the tank's body).

use crate::document::{Document, ItemValue, Obj, ObjectId};
use crate::error::{Diagnostic, Error, Location, Result};
use crate::rules::RuleScope;
use crate::text::py_str_repr;

/// The numbering of the copy of the enabled objects in the model: per body, lid and field point of
/// the copy its zero-based index in the model (the field points of the sea, then those of every tank
/// of the bodies in order), and the owners of the field points of the model: the points of the sea
/// and per tank its body, its path and its first point and count among them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Numbering {
    pub bodies: Vec<usize>,
    pub lids: Vec<usize>,
    pub field_points: Vec<usize>,
    pub sea_points: usize,
    pub tanks: Vec<TankPoints>,
}

/// The field points of one tank among those of a model: its body (zero-based, in the model), its
/// path ("bodies/hull/tanks/aft"), its first point (zero-based) and its count.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TankPoints {
    pub body: usize,
    pub path: String,
    pub first: usize,
    pub count: usize,
}

// the index in the model of an index of the copy (an index beyond the numbering kept)
fn of(v: &[usize], i: usize) -> usize {
    v.get(i).copied().unwrap_or(i)
}

// the text with "field point <a>" (not followed by a digit) as `b`
fn renumbered(text: &str, a: usize, b: &str) -> String {
    let from = format!("field point {a}");
    let mut out = String::new();
    let mut rest = text;
    while let Some(k) = rest.find(&from) {
        let end = k + from.len();
        let digit = rest[end..].chars().next().is_some_and(|c| c.is_ascii_digit());
        out += &rest[..k];
        out += if digit { &from } else { b };
        rest = &rest[end..];
    }
    out + rest
}

impl Numbering {
    // the owner of field point i of the copy: its zero-based index in its owner's list and its tank
    // (None: a point of the sea)
    fn owner(&self, i: usize) -> (usize, Option<&TankPoints>) {
        let k = of(&self.field_points, i);
        match self.tanks.iter().find(|t| k >= t.first && k < t.first + t.count) {
            Some(t) if k >= self.sea_points => (k - t.first, Some(t)),
            _ => (k, None),
        }
    }
    /// A location of the copy as a location of the model: a field point of a tank as its index in
    /// the tank's field_points, with the tank's body.
    pub fn location(&self, l: &Location) -> Location {
        let mut out = Location {
            body: l.body.map(|i| of(&self.bodies, i)),
            lid: l.lid.map(|i| of(&self.lids, i)),
            field_point: l.field_point.map(|i| self.owner(i).0),
            ..*l
        };
        if let Some((_, Some(t))) = l.field_point.map(|i| self.owner(i)) {
            out.body = Some(t.body);
        }
        out
    }
    // the message of a location of the copy: a field point named by its one-based number in its
    // owner's list, a point of a tank with the tank
    fn message(&self, l: &Location, m: &str) -> String {
        let Some(i) = l.field_point else { return m.to_string() };
        match self.owner(i) {
            (k, None) if k != i => renumbered(m, i + 1, &format!("field point {}", k + 1)),
            (_, None) => m.to_string(),
            (k, Some(t)) => renumbered(m, i + 1, &format!("field point {} of {}", k + 1, t.path)),
        }
    }
    /// An error of the copy as an error of the model.
    pub fn error(&self, e: Error) -> Error {
        let l = *e.location();
        let m = self.message(&l, e.message());
        e.with_message(m).at(self.location(&l))
    }
    /// A diagnostic of the copy as a diagnostic of the model.
    pub fn diagnostic(&self, d: &Diagnostic) -> Diagnostic {
        Diagnostic { location: self.location(&d.location), message: self.message(&d.location, &d.message), ..d.clone() }
    }
}

/// The copy of the enabled objects of a model and its numbering.
#[derive(Clone, Debug)]
pub struct Enabled {
    pub doc: Document,
    pub numbering: Numbering,
}

/// Whether an object takes part in the solve by its own item `enabled` (an object without the
/// item always does).
pub fn is_enabled(o: Obj) -> bool {
    o.type_().item("enabled").is_none() || o.boolean("enabled").expect("the schema gives enabled a default")
}

/// The problems built from a model, each from its own copy of the document (`scoped_document`):
/// - `Model`: the model as it is solved (its enabled objects; a check, the route, a solve);
/// - `Body(b)`: the queries of one body (b taken as enabled, its disabled children left out; the
///   other bodies and the lids placed with them gone, but for the bodies whose waterline loops a
///   generated plate of b covers; without the motions);
/// - `BodyMesh(b)`: the mesh of one body as the problem uses it (the enabled objects, b taken as
///   enabled, its disabled children left out; the other bodies kept);
/// - `FreeSurface`: the queries of the potential QTF's free surface (the enabled objects, every
///   body; without the motions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Model,
    Body(ObjectId),
    BodyMesh(ObjectId),
    FreeSurface,
}

impl Scope {
    /// The rules its problem is checked by (`Problem::check_scope`); None for a body's mesh, of
    /// which no problem is built.
    pub fn rules(self) -> Option<RuleScope> {
        match self {
            Scope::Model => Some(RuleScope::Model),
            Scope::Body(_) => Some(RuleScope::Body),
            Scope::BodyMesh(_) => None,
            Scope::FreeSurface => Some(RuleScope::FreeSurface),
        }
    }
}

/// Which objects the walk of the tree takes as enabled besides those enabled by their own item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Walk {
    /// None: the model as it is solved.
    Solve,
    /// The object and every object within it (the check of a disabled object, `Model::check`).
    Force(ObjectId),
    /// The object itself alone: the disabled objects within it stay suppressed (the queries of a
    /// body: the body as the model would solve it, were it enabled).
    Take(ObjectId),
}

/// The objects not enabled that are not within another one, in the order of the tree, but those
/// the walk takes as enabled.
pub fn suppressed(doc: &Document, walk: Walk) -> Vec<ObjectId> {
    fn go(o: Obj, walk: Walk, forced: bool, out: &mut Vec<ObjectId>) {
        let forced = forced || walk == Walk::Force(o.id());
        if !forced && walk != Walk::Take(o.id()) && !is_enabled(o) {
            out.push(o.id());
            return;
        }
        for l in &o.type_().lists {
            for c in o.children(l.name) {
                go(c, walk, forced, out);
            }
        }
    }
    let mut out = Vec::new();
    go(doc.root(), walk, false, &mut out);
    out
}

fn bool_value(b: bool) -> ItemValue {
    ItemValue::Bool(b)
}

// ErrorKind::Value when the walk suppresses every body of the model
fn some_body(doc: &Document, walk: Walk) -> Result<()> {
    let taken = |b: &Obj| matches!(walk, Walk::Force(id) | Walk::Take(id) if id == b.id());
    let bodies = doc.root().children("bodies");
    if !bodies.is_empty() && bodies.iter().all(|b| !is_enabled(*b) && !taken(b)) {
        return Err(Error::value(
            "every body is disabled: the model has no body to solve (set enabled = true on one of them)",
        ));
    }
    Ok(())
}

// the copy without the motions (`first_order.motion` false: the inertia is no input of its
// queries)
fn without_motions(mut e: Enabled) -> Result<Enabled> {
    let out = e.doc.root().child("first_order", "").expect("the first_order").id();
    e.doc.set(out, "motion", bool_value(false))?;
    Ok(e)
}

/// The copy of a model for a scope and its numbering. ErrorKind::Value when every body is
/// suppressed (not in a body's scope, which takes the body as enabled), and for a lid placed with
/// a suppressed body.
pub fn scoped_document(doc: &Document, scope: Scope) -> Result<Enabled> {
    match scope {
        Scope::Model => walked_document(doc, Walk::Solve),
        Scope::Body(body) => without_motions(body_document(doc, body)?),
        Scope::BodyMesh(body) => walked_document(doc, Walk::Take(body)),
        Scope::FreeSurface => without_motions(walked_document(doc, Walk::Solve)?),
    }
}

/// The copy of the objects of a model that the walk takes as enabled (`Walk::Solve`: the model's
/// scope; `Walk::Force(id)`: the check of the disabled object id as if it alone were enabled).
/// ErrorKind::Value as `scoped_document`.
pub fn walked_document(doc: &Document, walk: Walk) -> Result<Enabled> {
    some_body(doc, walk)?;
    copy_without(doc, suppressed(doc, walk))
}

// the copy of a model for the queries of one body (`body`, taken as enabled), the motions kept:
// its enabled objects without the other bodies and the lids placed with them. The queries of a
// body thus see the body, its surfaces, the lids placed with it, the fixed dampers and the
// environment, never the options, orders, frequencies or other bodies (reference/queries).
fn body_document(doc: &Document, body: ObjectId) -> Result<Enabled> {
    let root = doc.root();
    let own = doc.obj(body).name().to_string();
    // the objects gone of the model's lists (the other bodies with everything under them, the lids
    // placed with them), then those suppressed within the objects kept
    // (a body whose waterline loop a generated plate of the body covers is kept: the plate is made
    // from its trimmed mesh)
    let over: Vec<&str> = doc
        .obj(body)
        .children("free_surface_plates")
        .iter()
        .filter(|p| is_enabled(**p) && p.has("inside"))
        .filter_map(|p| p.body("body"))
        .collect();
    let mut gone = Vec::new();
    for b in root.children("bodies") {
        if b.id() != body && !over.contains(&b.name()) {
            gone.push(b.id());
        }
    }
    for l in root.children("lids") {
        if l.body("placed_with").is_some_and(|w| w != own) {
            gone.push(l.id());
        }
    }
    let under_gone = |id: ObjectId| {
        let mut o = Some(doc.obj(id));
        while let Some(x) = o {
            if gone.contains(&x.id()) {
                return true;
            }
            o = x.parent();
        }
        false
    };
    // (the body taken as enabled itself: its own disabled children stay out, so that its rows are
    // those of the model's queries)
    let within: Vec<ObjectId> = suppressed(doc, Walk::Take(body)).into_iter().filter(|&id| !under_gone(id)).collect();
    gone.extend(within);
    copy_without(doc, gone)
}

// the copy of a model without the objects gone (each with everything under it) and its numbering
fn copy_without(doc: &Document, gone: Vec<ObjectId>) -> Result<Enabled> {
    let root = doc.root();
    let bodies = root.children("bodies");
    let mut n = Numbering::default();
    let keep = |o: &Obj| !gone.contains(&o.id());
    n.bodies = bodies.iter().enumerate().filter(|(_, b)| keep(b)).map(|(k, _)| k).collect();
    let lids = root.children("lids");
    n.lids = lids.iter().enumerate().filter(|(_, l)| keep(l)).map(|(k, _)| k).collect();
    // the field points of the model in the order of the solve, those kept: of the sea, and of the
    // tanks kept (a tank kept with its body)
    let count =
        |o: Obj, item: &str| -> Result<usize> { Ok(if o.has(item) { o.shape(item)?[0].max(0) as usize } else { 0 }) };
    let first = root.child("first_order", "").expect("the model has its first_order");
    let sea = count(first, "field_points")?;
    let mut all = sea;
    n.field_points = (0..sea).collect();
    n.sea_points = sea;
    for (bi, b) in bodies.iter().enumerate() {
        for t in b.children("tanks") {
            let k = count(t, "field_points")?;
            if keep(b) && keep(&t) {
                n.field_points.extend(all..all + k);
            }
            if k > 0 {
                n.tanks.push(TankPoints { body: bi, path: t.path(), first: all, count: k });
            }
            all += k;
        }
    }
    if gone.is_empty() {
        return Ok(Enabled { doc: doc.clone(), numbering: n });
    }
    let mut d = doc.clone();
    d.keep_places();
    for &id in &gone {
        d.take_out(id);
    }
    // the couplings that name a suppressed body go with it
    let enabled_bodies: Vec<String> = d.root().children("bodies").iter().map(|b| b.name().to_string()).collect();
    let mut drop = Vec::new();
    for b in d.root().children("bodies") {
        for c in b.children("couplings") {
            if c.body("body").is_some_and(|v| !enabled_bodies.iter().any(|b| b == v)) {
                drop.push(c.id());
            }
        }
    }
    for id in drop {
        d.take_out(id);
    }
    // a plate generated over the waterline of a suppressed body covers nothing
    for b in d.root().children("bodies") {
        for p in b.children("free_surface_plates") {
            if let Some(v) = p.body("body").filter(|v| !enabled_bodies.iter().any(|b| b == v)) {
                return Err(Error::value(format!(
                    "{}body: the body {} is disabled (disable the plate as well)",
                    p.prefix(),
                    py_str_repr(v)
                )));
            }
        }
    }
    // a lid placed with a suppressed body is placed by nothing
    for (k, &li) in n.lids.iter().enumerate() {
        let l = d.root().children("lids")[k];
        if let Some(w) = l.body("placed_with").filter(|w| !enabled_bodies.iter().any(|b| b == w)) {
            return Err(Error::value(format!(
                "{}placed_with: the body {} is disabled (disable the lid as well, or give it in global coordinates)",
                l.prefix(),
                py_str_repr(w)
            ))
            .at(Location { lid: Some(li), ..Location::default() }));
        }
    }
    // the field outputs go when every field point went with its tank
    if all > 0 && n.field_points.is_empty() {
        let out = d.root().child("first_order", "").expect("the first_order").id();
        d.set(out, "field_pressure", bool_value(false))?;
        d.set(out, "field_velocity", bool_value(false))?;
    }
    Ok(Enabled { doc: d, numbering: n })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_point_is_named_by_its_number_in_the_model() {
        assert_eq!(
            renumbered("field point 1 lies inside x; field point 12", 1, "field point 3"),
            "field point 3 lies inside x; field point 12"
        );
        let n = Numbering { bodies: vec![1, 2], lids: vec![], field_points: vec![0, 4], sea_points: 5, tanks: vec![] };
        let at = Location { field_point: Some(1), body: Some(0), ..Location::default() };
        let e = n.error(Error::value("field point 2 is not finite").at(at));
        assert_eq!(e.message(), "field point 5 is not finite");
        assert_eq!((e.location().field_point, e.location().body), (Some(4), Some(1)));
    }

    #[test]
    fn a_field_point_of_a_tank_is_named_with_its_tank() {
        // two points in the sea, then three of a tank of the model's body 1 (the copy without body 0)
        let tank = TankPoints { body: 1, path: "bodies/hull/tanks/aft".into(), first: 2, count: 3 };
        let n = Numbering {
            bodies: vec![1],
            lids: vec![],
            field_points: vec![0, 1, 2, 3, 4],
            sea_points: 2,
            tanks: vec![tank],
        };
        let at = Location { field_point: Some(3), ..Location::default() };
        let e = n.error(Error::value("field point 4 lies above the free surface of its tank").at(at));
        assert_eq!(e.message(), "field point 2 of bodies/hull/tanks/aft lies above the free surface of its tank");
        assert_eq!((e.location().field_point, e.location().body), (Some(1), Some(1)));
    }
}
