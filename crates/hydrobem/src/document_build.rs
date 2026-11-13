//! The engine's problem of a model: the rules between the items of an object and between objects,
//! then the problem filled from the objects.
//!
//! Every surface names its geometry by its items mesh (a mesh of the model) and ranges (of its
//! stored elements; unset: every stored element). The surfaces of a body on the body's own mesh are
//! roles of the elements of that mesh (each element at most one), the others are meshes of their own.
//! A trimmed body is placed and cut at the build: its mesh is trimmed as the trim of a mesh does it
//! (`Model::trim_mesh`), and every range is mapped onto the elements that come from it. A body's
//! engine mesh holds the elements its surfaces take, in the stored order (the whole mesh when they
//! take every element). The generated control surfaces are made here as the engine made them as
//! meshes; the generated interior surfaces where the solve generates them; a detected interior
//! surface is the elements of the wetted surface lying in z = 0.

use crate::codes::{ControlSurfaceKind, QuantityId};
use crate::constants::STANDARD_GRAVITY;
use crate::document::{
    CONTROL_SURFACE, Document, FREE_SURFACE, INTERIOR_SURFACE, Obj, ObjectId, OriginItem, SurfaceItems, check_switched,
};
use crate::document_enabled::Scope;
use crate::document_mesh::{MeshDesc, fill_mesh, mesh_desc};
use crate::error::{Error, Location, Result, ResultExt, at_body, int32_of};
use crate::gdf::GdfFile;
use crate::geometry::{Placement, Vec3, reflections, v3};
use crate::ho_merge::merge_other_meshes;
use crate::mesh::{Mesh, Range, build_mesh, build_patch_mesh};
use crate::mesh_desc::{LowOrderMeshDesc, PatchDiscretization, PatchKind, PatchMeshDesc};
use crate::model::Model;
use crate::model_codes::{
    DamperCondition, Domain, Excitation, Formulation, InteriorMethod, Mass, NewmanDrifts, Origin, PressureShape,
    QTF_DIFFERENCES, QTF_SUMS, QtfFrequencies, QtfHeadings, RestoringConvention, WaterlineTerm,
};
use crate::model_meshes::{control_surface_gdf, control_surface_gdf_covered};
use crate::model_types::{ControlSurfaceInput, Trim};
use crate::problem::{
    Body, Damper, DipoleOwner, FixedDamper, GeneralizedMode, Lid, MeshOrigin, ModePiece, ModeTerm, Plate, PressureMode,
    PressureSurface, Problem, Tank,
};
use crate::problem::{plane_tol, some_index};
use crate::text::{element_text, py_str_repr};

const MODE_NAMES: [&str; 6] = ["surge", "sway", "heave", "roll", "pitch", "yaw"];
const DOMAINS: [(&str, Domain); 5] = [
    ("open_sea", Domain::OpenSea),
    ("x_wall", Domain::XWall),
    ("y_wall", Domain::YWall),
    ("orthogonal_walls", Domain::OrthogonalWalls),
    ("channel", Domain::Channel),
];
// the generated interior surfaces of the item Body.interior_surface
const INTERIOR_METHODS: [(&str, InteriorMethod); 3] = [
    ("projected", InteriorMethod::Projected),
    ("waterline", InteriorMethod::Waterline),
    ("graded", InteriorMethod::Graded),
];
const SHAPES: [(&str, PressureShape); 3] =
    [("uniform", PressureShape::Uniform), ("sign_x", PressureShape::SignX), ("sign_y", PressureShape::SignY)];
const INERTIA: [&str; 6] = ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"];

// the index of a choice the schema checked (a choice it does not list is a bug)
fn index_of(names: &[&str], s: &str) -> i32 {
    match names.iter().position(|n| *n == s) {
        Some(k) => k as i32,
        None => panic!("no choice '{s}'"),
    }
}

// the value of a choice the schema checked (a choice it does not list is a bug)
fn choice<T: Copy>(table: &[(&str, T)], s: &str) -> T {
    match table.iter().find(|(n, _)| *n == s) {
        Some(&(_, v)) => v,
        None => panic!("no choice '{s}'"),
    }
}

// the always-present single child of a list ("options", "environment")
fn single<'a>(o: Obj<'a>, list: &str) -> Obj<'a> {
    o.child(list, "").unwrap_or_else(|| panic!("the model has its {list}"))
}

// the method of a body's interior surface ("none": none)
fn interior_method<'a>(b: Obj<'a>) -> Result<&'a str> {
    b.choice("interior_surface")
}

// the method of a body's control surface ("none": none)
fn control_method<'a>(b: Obj<'a>) -> Result<&'a str> {
    b.choice("control_surface")
}

// ------------------------------------------------------------------ mesh references

/// The mesh of a surface: the name of a mesh and ranges of its stored elements (none: every element).
#[derive(Clone, Debug, PartialEq)]
pub struct MeshRef {
    pub name: String,
    pub ranges: Vec<Range>,
}

/// The ranges item of an object (none when it is unset: every element).
pub fn ranges_item(o: Obj) -> Vec<Range> {
    ranges_of_item(o, "ranges")
}

// the element ranges of a ranges item (none when it is unset: every element)
fn ranges_of_item(o: Obj, item: &str) -> Vec<Range> {
    let v = o.optional_ints(item).unwrap_or_default();
    v.chunks_exact(2).map(|c| Range { first: c[0] as i32, last: c[1] as i32 }).collect()
}

/// The mesh of a surface of flat items (its items `<prefix>mesh`, `<prefix>ranges`), None when it
/// has none.
fn prefixed_ref(o: Obj, prefix: &str) -> Option<MeshRef> {
    let name = o.mesh(&format!("{prefix}mesh"))?;
    Some(MeshRef { name: name.to_string(), ranges: ranges_of_item(o, &format!("{prefix}ranges")) })
}

impl MeshRef {
    /// The mesh of an object (its items mesh and ranges), None when it has no mesh.
    pub fn of(o: Obj) -> Option<MeshRef> {
        let name = o.mesh("mesh")?;
        Some(MeshRef { name: name.to_string(), ranges: ranges_item(o) })
    }
    /// The mesh of a surface whose method is "mesh" (a body's interior or control surface, the
    /// potential QTF's free surface): None under another method (its mesh then kept unused).
    pub fn given(o: Obj, s: SurfaceItems) -> Result<Option<MeshRef>> {
        Ok(if o.choice(s.choice)? == "mesh" { prefixed_ref(o, s.prefix) } else { None })
    }
    // the one-based elements it takes of a mesh of count elements
    fn records(&self, count: i32) -> Vec<i32> {
        if self.ranges.is_empty() {
            return (1..=count).collect();
        }
        self.ranges.iter().flat_map(|r| r.first..=r.last).collect()
    }
}

/// The ranges of sorted one-based elements (consecutive elements joined).
fn ranges_of(records: &[i32]) -> Vec<Range> {
    let mut out: Vec<Range> = Vec::new();
    for &r in records {
        match out.last_mut() {
            Some(x) if x.last + 1 == r => x.last = r,
            _ => out.push(Range { first: r, last: r }),
        }
    }
    out
}

/// The number of stored elements of a mesh descriptor (the elements of a trim, or the patches or
/// panels).
pub fn stored_records(d: &MeshDesc) -> i32 {
    if d.low_order {
        d.lo.panel_count
    } else if d.ho.trim_count > 0 {
        d.ho.trim_count
    } else {
        d.ho.patch_count
    }
}

// the descriptor of a mesh object of the model by name
fn desc_of(doc: &Document, name: &str) -> Result<MeshDesc> {
    let Some(o) = doc.root().child("meshes", name) else {
        return Err(Error::value(format!("unknown mesh {}", py_str_repr(name))));
    };
    mesh_desc(o)
}

/// The descriptor with only the stored elements keep[k] (zero-based) of it, in their order.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn keep_records(d: &mut MeshDesc, keep: &[bool]) {
    let rows = |v: &[f64], n: usize| -> Vec<f64> {
        let mut out = Vec::new();
        for (k, c) in v.chunks_exact(n).enumerate() {
            if keep[k] {
                out.extend_from_slice(c);
            }
        }
        out
    };
    let kept = keep.iter().filter(|&&k| k).count() as i32;
    if d.low_order {
        d.lo.vertices = rows(&d.lo.vertices, 12);
        d.lo.panel_count = kept;
        return;
    }
    let h = &mut d.ho;
    if h.trim_count > 0 {
        h.trim = h.trim.iter().zip(keep).filter(|(_, k)| **k).map(|(t, _)| *t).collect();
        h.trim_count = kept;
        return;
    }
    h.discretization = h.discretization.iter().zip(keep).filter(|(_, k)| **k).map(|(q, _)| *q).collect();
    match h.geometry {
        PatchKind::Flat => h.vertices = rows(&h.vertices, 12),
        PatchKind::Analytic => {
            h.analytic_patches = h.analytic_patches.iter().zip(keep).filter(|(_, k)| **k).map(|(a, _)| *a).collect()
        }
        PatchKind::Spline => {
            let (mut orders, mut counts, mut knots, mut coef) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut weights = Vec::new();
            let (mut kn, mut cf) = (0usize, 0usize);
            for p in 0..h.patch_count as usize {
                let (ku, kv) = (h.spline_orders[2 * p], h.spline_orders[2 * p + 1]);
                let (mu, mv) = (h.spline_counts[2 * p], h.spline_counts[2 * p + 1]);
                let nk = (mu + ku + mv + kv) as usize;
                let nc = 3 * (mu * mv) as usize;
                if keep[p] {
                    orders.extend_from_slice(&[ku, kv]);
                    counts.extend_from_slice(&[mu, mv]);
                    knots.extend_from_slice(&h.knots[kn..kn + nk]);
                    coef.extend_from_slice(&h.coefficients[cf..cf + nc]);
                    if !h.weights.is_empty() {
                        weights.extend_from_slice(&h.weights[cf / 3..(cf + nc) / 3]);
                    }
                }
                kn += nk;
                cf += nc;
            }
            h.spline_orders = orders;
            h.spline_counts = counts;
            h.knots = knots;
            h.knot_count = h.knots.len() as i64;
            h.coefficients = coef;
            h.coefficient_count = (h.coefficients.len() / 3) as i64;
            h.weights = weights;
        }
    }
    h.patch_count = kept;
}

// the engine's mesh of a descriptor (the gravity of a mesh file is not used)
pub(crate) fn engine_mesh(mut d: MeshDesc, index: usize) -> Result<Mesh> {
    d.lo.gravity = STANDARD_GRAVITY;
    d.ho.gravity = STANDARD_GRAVITY;
    let at = Location { mesh: Some(index), ..Location::default() };
    if d.low_order { build_mesh(&d.lo).located(at) } else { build_patch_mesh(&d.ho).located(at) }
}

/// The descriptor of a mesh the engine makes (a GdfFile), as it is when kept as a mesh object of a
/// model and read back.
pub fn gdf_desc(g: &GdfFile) -> Result<MeshDesc> {
    let mut doc = Document::new();
    let root = doc.root_id();
    let id = doc.add_child(root, "meshes", "Mesh", "made", -1)?;
    fill_mesh(&mut doc.obj_mut(id), g, false)?;
    mesh_desc(doc.obj(id))
}

// ------------------------------------------------------------------ the roles of a body's mesh

/// What an element of a body's mesh is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Wet,
    Dipole(usize),
    Damper(usize),
    Tank(usize),
    TankDipole(usize, usize),
    TankDamper(usize, usize),
    TankPlate(usize, usize),
    Pressure(usize),
    Plate(usize),
    Interior,
    Lid(usize),
    FixedDamper(usize),
}

impl Part {
    // the part in the messages: an object of a list by its name, or by its one-based position
    // without one (b: the body)
    fn text(self, b: Obj) -> String {
        let root = || b.document().root();
        let of = |o: Obj, list: &str, k: usize| -> String {
            let c = o.children(list)[k];
            element_text(list, c.position(), c.name())
        };
        let tank = |t: usize| b.children("tanks")[t];
        match self {
            Part::Wet => "the wetted surface (mesh)".to_string(),
            Part::Dipole(k) => of(b, "dipoles", k),
            Part::Damper(k) => of(b, "dipole_dampers", k),
            Part::Tank(k) => of(b, "tanks", k),
            Part::TankDipole(t, k) => format!("{} {}", of(b, "tanks", t), of(tank(t), "dipoles", k)),
            Part::TankDamper(t, k) => format!("{} {}", of(b, "tanks", t), of(tank(t), "dipole_dampers", k)),
            Part::TankPlate(t, k) => format!("{} {}", of(b, "tanks", t), of(tank(t), "free_surface_plates", k)),
            Part::Pressure(k) => of(b, "pressure_surfaces", k),
            Part::Plate(k) => of(b, "free_surface_plates", k),
            Part::Interior => "interior_surface".to_string(),
            Part::Lid(k) => format!("lids/{}", root().children("lids")[k].name()),
            Part::FixedDamper(k) => of(root(), "fixed_dampers", k),
        }
    }
}

/// The engine mesh of a body: its elements (after a trim), the stored element of the body's mesh each
/// comes from (0: an element the trim made, the interior surface), and the elements of every part.
pub struct BodyMesh {
    pub own: String, // the body's mesh: its item mesh, or the mesh of its first dipole or first plate
    pub desc: MeshDesc,
    pub source: Vec<i32>, // per engine element the one-based stored element, 0 for a generated one
    pub copy: Vec<[i32; 2]>, // per engine element the copy (sx, sy) of the stored element it was cut from
    pub part: Vec<i32>,   // per engine element its part among the pieces of that copy (0: not cut)
    pub parts: Vec<(Part, MeshRef)>,
    pub stored: i32,
    pub trimmed: bool,
    pub plate: Vec<i32>, // per engine element the plate a generated element is part of (-1: none)
}

impl BodyMesh {
    /// The one-based engine elements that come from the stored elements of a reference.
    pub fn records(&self, r: &MeshRef) -> Vec<i32> {
        let take: std::collections::BTreeSet<i32> = r.records(self.stored).into_iter().collect();
        let mut out = Vec::new();
        for (k, &s) in self.source.iter().enumerate() {
            if take.contains(&s) {
                out.push(k as i32 + 1);
            }
        }
        out
    }
    /// The engine elements of a part as ranges (of a plate also the elements generated for it).
    pub fn ranges(&self, part: Part) -> Vec<Range> {
        let mut recs = Vec::new();
        for (p, r) in &self.parts {
            if *p == part {
                recs.extend(self.records(r));
            }
        }
        if let Part::Plate(k) = part {
            recs.extend((0..self.plate.len()).filter(|&e| self.plate[e] == k as i32).map(|e| e as i32 + 1));
        }
        recs.sort();
        ranges_of(&recs)
    }
    /// The engine elements the trim generated for no plate (the interior surface of a trimmed patch
    /// mesh).
    pub fn generated(&self) -> Vec<Range> {
        let recs: Vec<i32> = (0..self.source.len())
            .filter(|&k| self.source[k] == 0 && self.plate.get(k).is_none_or(|&p| p < 0))
            .map(|k| k as i32 + 1)
            .collect();
        ranges_of(&recs)
    }
}

/// Whether a free-surface plate is generated from a waterline loop (its item inside) instead of
/// given by a mesh.
pub fn generated_plate(p: Obj) -> bool {
    p.has("inside")
}

// the point (x, y) of a generated plate
fn inside_of(p: Obj) -> Result<[f64; 2]> {
    let v = p.reals("inside")?;
    Ok([v[0], v[1]])
}

/// The body whose waterline loop a generated plate of body b covers: the body its item body names,
/// else b.
fn waterline_body<'a>(b: Obj<'a>, p: Obj<'a>) -> &'a str {
    p.body("body").unwrap_or(b.name())
}

/// The plates of a body generated over its own trimmed waterline: (their place in
/// free_surface_plates, the point inside), in order (the plates of `Trim::plates`).
pub fn own_generated_plates(b: Obj) -> Result<Vec<(usize, [f64; 2])>> {
    let mut out = Vec::new();
    for (k, p) in b.children("free_surface_plates").iter().enumerate() {
        if generated_plate(*p) && waterline_body(b, *p) == b.name() {
            out.push((k, inside_of(*p)?));
        }
    }
    Ok(out)
}

// a point item (x, y, z)
fn point(o: Obj, item: &str) -> Result<Vec3> {
    let v = o.reals(item)?;
    Ok(v3(v[0], v[1], v[2]))
}

// a body of generated plates only: no mesh, no dipoles, its first plate generated (over another
// body's waterline: the rules refuse its own)
fn of_generated_plates(b: Obj) -> bool {
    !b.has("mesh")
        && b.children("dipoles").is_empty()
        && b.children("free_surface_plates").first().is_some_and(|p| generated_plate(*p))
}

/// The mesh of the generated plate k of body b over the waterline loop of another body (a piston
/// body over a trimmed tube), in the coordinates of b: the triangles of
/// preprocess::mesh_plate_triangles (low order, panels) or the flat patches of
/// preprocess::mesh_plate_patches (higher order) on the engine mesh of the other body (trimmed),
/// moved from its body axes into those of b. In the stored symmetry form of the other body's mesh
/// when the two bodies have the same position and yaw, else every copy about its planes (no
/// symmetry).
fn other_plate_desc(b: Obj, k: usize) -> Result<MeshDesc> {
    let p = b.children("free_surface_plates")[k];
    let at = p.prefix();
    let doc = b.document();
    let name = waterline_body(b, p);
    let Some(o) = doc.root().children("bodies").into_iter().find(|x| x.name() == name) else {
        return Err(Error::value(format!("{at}body: no body {}", py_str_repr(name))));
    };
    let placed = placed_fixed_dampers(doc)?;
    let om = body_mesh(o, &placed[o.index()])?;
    let m = engine_mesh(om.desc.clone(), 0)?;
    let named = |e: Error| {
        let msg = format!("{at}{}", e.message());
        e.with_message(msg)
    };
    let (po, yo) = (point(o, "position")?, o.real("yaw")?);
    let (pb, yb) = (point(b, "position")?, b.real("yaw")?);
    let same = po == pb && yo == yb;
    let copies: Vec<(f64, f64)> =
        if same { vec![(1.0, 1.0)] } else { reflections(m.isx, m.isy).iter().map(|r| (r.sx, r.sy)).collect() };
    let (sym_x, sym_y) = if same { (m.isx, m.isy) } else { (0, 0) };
    // the other body's coordinates into b's: x_b = R(-yaw_b) (p_o + R(yaw_o) x_o - p_b)
    let yaw = (yb.rem_euclid(360.0)).to_radians();
    let (c, s) = (yaw.cos(), yaw.sin());
    let from = Placement { position: po, yaw_deg: yo };
    let place = |x: Vec3| -> Vec3 {
        if same {
            return x;
        }
        let d = from.apply(x) - pb;
        v3(c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2])
    };
    // a copy of a corner (sx, sy) and the orientation of a reflection of odd order restored
    let copy = |q: &[Vec3], sx: f64, sy: f64| -> Vec<Vec3> {
        let mut v: Vec<Vec3> = q.iter().map(|x| place(v3(sx * x[0], sy * x[1], x[2]))).collect();
        if sx * sy < 0.0 {
            v[1..].reverse();
        }
        v
    };
    let mut d = MeshDesc { label: format!("the plate over {}", py_str_repr(name)), ..MeshDesc::default() };
    if !m.patch {
        let tris = crate::preprocess::mesh_plate_triangles(&m, inside_of(p)?).map_err(named)?;
        let mut v = Vec::new();
        for &(sx, sy) in &copies {
            for t in &tris {
                // the normal down, into the water
                let q = copy(&[t[0], t[2], t[1]], sx, sy);
                for x in [q[0], q[1], q[2], q[2]] {
                    v.extend_from_slice(&x.0);
                }
            }
        }
        d.lo = LowOrderMeshDesc {
            panel_count: (v.len() / 12) as i32,
            vertices: v,
            reference_length: m.reference_length,
            gravity: STANDARD_GRAVITY,
            symmetry_x: sym_x,
            symmetry_y: sym_y,
        };
        return Ok(d);
    }
    let quads = crate::preprocess::mesh_plate_patches(&m, inside_of(p)?).map_err(named)?;
    let d0 = m.discretization[0];
    let mut v = Vec::new();
    let mut disc = Vec::new();
    for &(sx, sy) in &copies {
        for (q, nu, nv) in &quads {
            for x in copy(q, sx, sy) {
                v.extend_from_slice(&x.0);
            }
            // the reversed corners exchange u and v
            let (nu, nv) = if sx * sy < 0.0 { (*nv, *nu) } else { (*nu, *nv) };
            disc.push(PatchDiscretization { nu, nv, ..d0 });
        }
    }
    d.low_order = false;
    d.ho = PatchMeshDesc {
        geometry: PatchKind::Flat,
        patch_count: disc.len() as i32,
        vertices: v,
        discretization: disc,
        reference_length: m.reference_length,
        gravity: STANDARD_GRAVITY,
        symmetry_x: sym_x,
        symmetry_y: sym_y,
        ..PatchMeshDesc::default()
    };
    Ok(d)
}

// the trim of a body: elevation, pitch, roll
fn trim_of(b: Obj) -> Option<[f64; 3]> {
    let t = b.optional_reals("trim")?;
    Some([t[0], t[1], t[2]])
}

fn higher_order(doc: &Document) -> Result<bool> {
    single(doc.root(), "options").boolean("higher_order")
}

/// The name of the mesh of a body: its item mesh, or for a body without a wetted surface the mesh
/// of its first dipole (a body of sheets), else of its first free-surface plate (a body of plates);
/// None for a body with none of them (refused by check_objects).
pub fn body_mesh_name(b: Obj) -> Option<String> {
    if let Some(r) = MeshRef::of(b) {
        return Some(r.name);
    }
    if let Some(d) = b.children("dipoles").first() {
        return MeshRef::of(*d).map(|r| r.name);
    }
    b.children("free_surface_plates").first().and_then(|p| MeshRef::of(*p)).map(|r| r.name)
}

/// The parts of a body on its own mesh (body_mesh_name), in order: the wetted surface, then the
/// dipoles, dipole dampers, tanks with their sheets, pressure surfaces, free-surface plates and the
/// interior surface that name it, the lids placed with the body and the fixed dampers placed in its
/// frame.
pub fn body_parts(b: Obj, fixed_dampers: &[usize]) -> Result<Vec<(Part, MeshRef)>> {
    let own = body_mesh_name(b).expect("a body has its mesh or a dipole (check_objects)");
    let mut out = Vec::new();
    if let Some(r) = MeshRef::of(b) {
        out.push((Part::Wet, r));
    }
    let on = |r: Option<MeshRef>| r.filter(|r| r.name == own);
    for (k, x) in b.children("dipoles").iter().enumerate() {
        if let Some(r) = on(MeshRef::of(*x)) {
            out.push((Part::Dipole(k), r));
        }
    }
    for (k, x) in b.children("dipole_dampers").iter().enumerate() {
        if let Some(r) = on(MeshRef::of(*x)) {
            out.push((Part::Damper(k), r));
        }
    }
    for (k, x) in b.children("tanks").iter().enumerate() {
        if let Some(r) = on(MeshRef::of(*x)) {
            out.push((Part::Tank(k), r));
        }
        for (j, d) in x.children("dipoles").iter().enumerate() {
            if let Some(r) = on(MeshRef::of(*d)) {
                out.push((Part::TankDipole(k, j), r));
            }
        }
        for (j, d) in x.children("dipole_dampers").iter().enumerate() {
            if let Some(r) = on(MeshRef::of(*d)) {
                out.push((Part::TankDamper(k, j), r));
            }
        }
        for (j, d) in x.children("free_surface_plates").iter().enumerate() {
            if let Some(r) = on(MeshRef::of(*d)) {
                out.push((Part::TankPlate(k, j), r));
            }
        }
    }
    for (k, x) in b.children("pressure_surfaces").iter().enumerate() {
        if let Some(r) = on(MeshRef::of(*x)) {
            out.push((Part::Pressure(k), r));
        }
    }
    for (k, x) in b.children("free_surface_plates").iter().enumerate() {
        if let Some(r) = on(MeshRef::of(*x)) {
            out.push((Part::Plate(k), r));
        }
    }
    if let Some(r) = on(MeshRef::given(b, INTERIOR_SURFACE)?) {
        out.push((Part::Interior, r));
    }
    let root = b.document().root();
    for (k, l) in root.children("lids").iter().enumerate() {
        if l.body("placed_with") == Some(b.name()) {
            if let Some(r) = MeshRef::of(*l) {
                out.push((Part::Lid(k), r));
            }
        }
    }
    let dampers = root.children("fixed_dampers");
    for &k in fixed_dampers {
        if let Some(r) = MeshRef::of(dampers[k]) {
            out.push((Part::FixedDamper(k), r));
        }
    }
    Ok(out)
}

/// The fixed dampers placed in the frame of each body: a fixed damper that takes elements of the
/// mesh of a body at the origin without yaw or trim is a damper of that body's mesh (at rest, in no
/// force), the first such body; the others are sheets of their own.
pub fn placed_fixed_dampers(doc: &Document) -> Result<Vec<Vec<usize>>> {
    let root = doc.root();
    let bodies = root.children("bodies");
    let mut out = vec![Vec::new(); bodies.len()];
    for (k, d) in root.children("fixed_dampers").iter().enumerate() {
        let Some(r) = MeshRef::of(*d) else { continue };
        for (bi, b) in bodies.iter().enumerate() {
            let p = b.reals("position")?;
            let at_origin = p.iter().all(|&x| x == 0.0) && b.real("yaw")? == 0.0 && !b.has("trim");
            if at_origin && MeshRef::of(*b).is_some_and(|m| m.name == r.name) {
                out[bi].push(k);
                break;
            }
        }
    }
    Ok(out)
}

/// The engine mesh of a body: its mesh (trimmed when the body has a trim), with the elements its
/// parts take; ranges beyond the mesh and elements of two parts are refused.
pub fn body_mesh(b: Obj, fixed_dampers: &[usize]) -> Result<BodyMesh> {
    let at = b.prefix();
    let doc = b.document();
    let here = at_body(b.index() as i32);
    if of_generated_plates(b) {
        // a body of generated plates only: its mesh the first plate's, over another body's
        // waterline loop (the elements numbered from 1)
        let desc = other_plate_desc(b, 0).located(here)?;
        let n = stored_records(&desc);
        return Ok(BodyMesh {
            own: String::new(),
            desc,
            source: (1..=n).collect(),
            copy: Vec::new(),
            part: Vec::new(),
            parts: Vec::new(),
            stored: n,
            trimmed: false,
            plate: vec![0; n as usize],
        });
    }
    let mut parts = body_parts(b, fixed_dampers)?;
    let own = body_mesh_name(b).expect("a body has its mesh or a dipole (check_objects)");
    let mut desc = desc_of(doc, &own).located(here)?;
    let stored = stored_records(&desc);
    // the ranges count the stored elements; each element of the mesh has at most one part
    let mut owner: Vec<Option<Part>> = vec![None; stored.max(0) as usize];
    for (p, r) in &parts {
        // a lid placed with the body is located by its index too
        let place = match p {
            Part::Lid(k) => Location { lid: Some(*k), ..here },
            _ => here,
        };
        for x in &r.ranges {
            if x.last > stored {
                return Err(Error::value(format!(
                    "{at}{}: the element range ({}, {}) is beyond the mesh {} ({} elements)",
                    p.text(b),
                    x.first,
                    x.last,
                    py_str_repr(&own),
                    stored
                ))
                .at(place));
            }
        }
        for i in r.records(stored) {
            let k = (i - 1) as usize;
            if let Some(q) = owner[k] {
                return Err(Error::value(format!(
                    "{at}element {i} of the mesh {} is taken by both {} and {} (an element has one role in a body)",
                    py_str_repr(&own),
                    q.text(b),
                    p.text(b)
                ))
                .at(Location { element: Some(i as usize), ..place }));
            }
            owner[k] = Some(*p);
        }
    }
    // the interior surface detected: the elements of the wetted surface lying in z = 0 (not of a
    // trimmed body, whose hull the trim cuts at the waterline)
    if !b.has("trim") && interior_method(b)? == "detect" {
        let wet: Vec<i32> = (1..=stored).filter(|&i| owner[(i - 1) as usize] == Some(Part::Wet)).collect();
        let found = in_free_surface(&desc, &wet, b.reals("position")?[2]);
        if !found.is_empty() && found.len() < wet.len() {
            for &i in &found {
                owner[(i - 1) as usize] = Some(Part::Interior);
            }
            let rest: Vec<i32> = wet.iter().copied().filter(|i| !found.contains(i)).collect();
            for (p, r) in parts.iter_mut() {
                if *p == Part::Wet {
                    r.ranges = ranges_of(&rest);
                }
            }
            parts.push((Part::Interior, MeshRef { name: own.clone(), ranges: ranges_of(&found) }));
        }
    }
    let mut source: Vec<i32> = (1..=stored).collect();
    let mut copy: Vec<[i32; 2]> = Vec::new();
    let mut part: Vec<i32> = Vec::new();
    let mut plate: Vec<i32> = vec![-1; stored.max(0) as usize];
    let mut trimmed = false;
    if let Some(t) = trim_of(b) {
        trimmed = true;
        // the trim of the mesh, by the engine's trim of a mesh: the tanks with a free surface given
        // cut at it (low order), the interior surface generated inside the new waterline (higher
        // order, the method waterline)
        // (the sheets of a tank with its walls)
        let mut tanks = Vec::new();
        for (k, t) in b.children("tanks").iter().enumerate() {
            let Some(fs) = t.real_or_name("surface_level")? else { continue };
            let mut recs: Vec<i32> = Vec::new();
            for (p, r) in &parts {
                if matches!(p, Part::Tank(j) | Part::TankDipole(j, _) | Part::TankDamper(j, _) | Part::TankPlate(j, _) if *j == k)
                {
                    recs.extend(r.records(stored));
                }
            }
            if !recs.is_empty() {
                recs.sort();
                tanks.push((recs, fs));
            }
        }
        let interior = higher_order(doc)? && interior_method(b)? == "waterline";
        let mut scratch = Model::with_document(doc.clone());
        let mesh = doc.root().child("meshes", &own).expect("the mesh of a body").id();
        let mut name = format!("{own}_trimmed");
        while doc.root().child("meshes", &name).is_some() {
            name += "_";
        }
        // the plates generated over the new waterline (their elements after those of the trim)
        let generated = own_generated_plates(b)?;
        let plates = generated.iter().map(|g| g.1).collect();
        let trim = Trim { elevation: t[0], pitch: t[1], roll: t[2], interior_surface: interior, tanks, plates };
        let (id, copies, tags): (ObjectId, Vec<[i32; 2]>, Vec<i32>) = scratch
            .trim_mesh_copies(mesh, &name, &trim)
            .map_err(|e| {
                let m = format!("{at}trim: {}", e.message());
                e.with_message(m)
            })
            .located(here)?;
        let o = scratch.document().obj(id);
        desc = mesh_desc(o)?;
        source = o.ints("source_elements")?.iter().map(|&s| if s > stored as i64 { 0 } else { s as i32 }).collect();
        // the part of an element: its place among the pieces of the same copy of its stored
        // element, in the order of the trim (an element the trim generated: 0)
        let mut seen: std::collections::HashMap<(i32, [i32; 2]), i32> = std::collections::HashMap::new();
        for (k, &s) in source.iter().enumerate() {
            let c = copies.get(k).copied().unwrap_or([1, 1]);
            let n = seen.entry((s, c)).or_insert(0);
            part.push(if s == 0 { 0 } else { *n });
            *n += 1;
        }
        copy = source.iter().enumerate().map(|(k, _)| copies.get(k).copied().unwrap_or([1, 1])).collect();
        plate = (0..source.len())
            .map(|k| match tags.get(k) {
                Some(&t) if t >= 0 => generated[t as usize].0 as i32,
                _ => -1,
            })
            .collect();
    }
    // the elements the parts take (and those the trim generated)
    let keep: Vec<bool> = source.iter().map(|&s| s == 0 || owner[(s - 1) as usize].is_some()).collect();
    if keep.iter().any(|k| !k) {
        keep_records(&mut desc, &keep);
        let kept = |v: &[i32]| v.iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| *s).collect::<Vec<_>>();
        if trimmed {
            part = kept(&part);
            copy = copy.iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| *s).collect();
        }
        source = kept(&source);
        plate = kept(&plate);
    }
    Ok(BodyMesh { own, desc, source, copy, part, parts, stored, trimmed, plate })
}

/// The elements (one-based, of `elements`) of a mesh descriptor lying wholly in the free surface
/// z = 0 when the mesh is placed at the elevation z0: every point of an element's hull within the
/// tolerance of the plane (the test of the model rules for a wetted element in z = 0). None when the
/// elements do not make a mesh (the build reports that).
fn in_free_surface(d: &MeshDesc, records: &[i32], z0: f64) -> Vec<i32> {
    if records.is_empty() {
        return Vec::new();
    }
    let n = stored_records(d);
    let mut keep = vec![false; n.max(0) as usize];
    for &i in records {
        keep[(i - 1) as usize] = true;
    }
    let mut part = d.clone();
    keep_records(&mut part, &keep);
    let Ok(m) = engine_mesh(part, 0) else { return Vec::new() };
    let mut out = Vec::new();
    for (k, &i) in records.iter().enumerate() {
        let Some(sh) = m.shapes.get(k) else { break };
        let tol = plane_tol(&m, sh);
        if !sh.hull.is_empty() && sh.hull.iter().all(|v| (v[2] + z0).abs() <= tol) {
            out.push(i);
        }
    }
    out
}

/// The one role of an element (reference/inputs-meshes §3) on the meshes of a body other than its own (own): the
/// elements of another mesh that its dipoles, its pressure surfaces, its free-surface plates and its
/// interior surface take are taken once (a range beyond its mesh is refused where the mesh is made).
fn other_roles(b: Obj, own: &str) -> Result<()> {
    let at = b.prefix();
    let mut parts: Vec<(Part, MeshRef)> = Vec::new();
    let mut add = |p: Part, r: Option<MeshRef>| {
        if let Some(r) = r.filter(|r| r.name != own) {
            parts.push((p, r));
        }
    };
    // in the order of body_parts
    for (k, x) in b.children("dipoles").iter().enumerate() {
        add(Part::Dipole(k), MeshRef::of(*x));
    }
    for (k, x) in b.children("pressure_surfaces").iter().enumerate() {
        add(Part::Pressure(k), MeshRef::of(*x));
    }
    for (k, x) in b.children("free_surface_plates").iter().enumerate() {
        add(Part::Plate(k), MeshRef::of(*x));
    }
    for (t, x) in b.children("tanks").iter().enumerate() {
        for (k, p) in x.children("free_surface_plates").iter().enumerate() {
            add(Part::TankPlate(t, k), MeshRef::of(*p));
        }
    }
    add(Part::Interior, MeshRef::given(b, INTERIOR_SURFACE)?);
    let mut owners: std::collections::BTreeMap<&str, Vec<Option<Part>>> = std::collections::BTreeMap::new();
    for (p, r) in &parts {
        if !owners.contains_key(r.name.as_str()) {
            let n = stored_records(&desc_of(b.document(), &r.name)?);
            owners.insert(&r.name, vec![None; n.max(0) as usize]);
        }
        let owner = owners.get_mut(r.name.as_str()).expect("added above");
        for i in r.records(owner.len() as i32) {
            let Some(o) = owner.get_mut((i - 1) as usize) else { continue };
            if let Some(q) = *o {
                return Err(Error::value(format!(
                    "{at}element {i} of the mesh {} is taken by both {} and {} (an element has one role in a body)",
                    py_str_repr(&r.name),
                    q.text(b),
                    p.text(b)
                )));
            }
            *o = Some(*p);
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ the rules between items and objects

// the mesh of a surface, which it needs
fn need_mesh(o: Obj, what: &str) -> Result<()> {
    if !o.has("mesh") {
        return Err(Error::value(format!("{}a {what} needs its mesh", o.prefix())));
    }
    Ok(())
}

fn check_body(b: Obj) -> Result<()> {
    let at = b.prefix();
    let dipoles = b.children("dipoles");
    for d in &dipoles {
        need_mesh(*d, "Dipole")?;
    }
    if !b.has("mesh") && b.has("ranges") {
        return Err(Error::value(format!("{at}ranges are given without the mesh")));
    }
    let plates = b.children("free_surface_plates");
    let trimmed = b.has("trim");
    if trimmed && !b.has("mesh") && dipoles.is_empty() && !plates.is_empty() {
        return Err(Error::value(format!(
            "{at}a body of free-surface plates only cannot be trimmed (any angle tilts the plates out of z = 0): \
             generate the plate over the trimmed hull's waterline instead (inside and body of a FreeSurfacePlate)"
        )));
    }
    for p in &plates {
        check_plate(b, *p, trimmed)?;
    }
    if !b.has("mesh") && dipoles.is_empty() && plates.is_empty() {
        // a body of sheets only has dipoles, one of plates only free-surface plates; a porous sheet
        // at rest is a fixed damper
        return Err(Error::value(format!("{at}the body has no wetted surface, no dipoles and no free-surface plates")));
    }
    // the inertia enters the motions of the rigid modes only (inertia_input chooses the items
    // read; both have values)
    let inertia = single(b.document().root(), "first_order").boolean("motion")? && !b.texts("modes")?.is_empty();
    if !b.has("center_of_gravity") {
        let coupled = |m: &str| b.children("couplings").iter().any(|c| c.has(m));
        let uses: [(&str, bool); 3] = [
            ("inertia_origin", inertia),
            ("stiffness_origin", b.has("additional_stiffness") || coupled("stiffness")),
            ("damping_origin", b.has("additional_damping") || coupled("damping")),
        ];
        for (origin, used) in uses {
            if used && b.origin(origin)? == OriginItem::Named("center_of_gravity") {
                return Err(Error::value(format!("{at}{origin}='center_of_gravity' requires center_of_gravity")));
            }
        }
    }
    check_couplings(b)?;
    let n = 6 + b.children("generalized_modes").len() as i64;
    for k in ["mode_mass", "mode_damping", "mode_stiffness"] {
        if !b.has(k) {
            continue;
        }
        let shape = b.shape(k)?;
        if shape[0] != n || shape[1] != n {
            return Err(Error::value(format!(
                "{at}{k} must be {n} x {n} (6 + G, G = {} generalized modes), not {} x {}",
                n - 6,
                shape[0],
                shape[1]
            )));
        }
    }
    let own = body_mesh_name(b).unwrap_or_default(); // none: a body of generated plates only
    check_switched(b)?;
    if trimmed && interior_method(b)? == "mesh" {
        return Err(Error::value(format!(
            "{at}a trimmed body has a generated interior surface (given elements lie on the waterline of the untrimmed hull): set interior_surface to a generated one",
        )));
    }
    for d in dipoles {
        if trimmed && MeshRef::of(d).is_some_and(|r| r.name != own) {
            return Err(Error::value(format!(
                "{}the dipoles of a trimmed body take elements of its mesh {}",
                d.prefix(),
                py_str_repr(&own)
            )));
        }
    }
    for d in b.children("dipole_dampers") {
        need_mesh(d, "DipoleDamper")?;
        if MeshRef::of(d).is_some_and(|r| r.name != own) {
            return Err(Error::value(format!(
                "{}a dipole damper of a body takes elements of the body's mesh {}",
                d.prefix(),
                py_str_repr(&own)
            )));
        }
    }
    for t in b.children("tanks") {
        need_mesh(t, "Tank")?;
        if MeshRef::of(t).is_some_and(|r| r.name != own) {
            return Err(Error::value(format!(
                "{}a tank takes elements of the body's mesh {}",
                t.prefix(),
                py_str_repr(&own)
            )));
        }
        // the sheets inside the tank: elements of the body's mesh in the tank fluid
        for (list, what) in [("dipoles", "Dipole"), ("dipole_dampers", "DipoleDamper")] {
            for d in t.children(list) {
                need_mesh(d, what)?;
                if MeshRef::of(d).is_some_and(|r| r.name != own) {
                    return Err(Error::value(format!(
                        "{}a sheet of a tank takes elements of the body's mesh {}",
                        d.prefix(),
                        py_str_repr(&own)
                    )));
                }
            }
        }
        // the plates of its free surface: elements of the body's mesh or a mesh of their own
        for p in t.children("free_surface_plates") {
            // given as a mesh: a plate generated in a loop of a trimmed waterline is a plate of the sea
            if p.has("inside") || p.has("body") {
                return Err(Error::value(format!(
                    "{}a plate of a tank's free surface is given by its mesh (inside and body generate a plate of \
                     the sea on a trimmed hull)",
                    p.prefix()
                )));
            }
            need_mesh(p, "FreeSurfacePlate")?;
            if trimmed {
                return Err(Error::value(format!(
                    "{}free-surface plates of a tank of a trimmed body are not available",
                    p.prefix()
                )));
            }
        }
    }
    for g in b.children("generalized_modes") {
        if g.children("pieces").is_empty() {
            return Err(Error::value(format!("{}a GeneralizedMode needs at least one ModePiece", g.prefix())));
        }
    }
    for p in b.children("pressure_surfaces") {
        need_mesh(p, "PressureSurface")?;
        if p.children("pressure_modes").is_empty() {
            return Err(Error::value(format!("{}a PressureSurface needs at least one PressureMode", p.prefix())));
        }
        if trimmed {
            return Err(Error::value(format!("{}a pressure surface of a trimmed body is not available", p.prefix())));
        }
    }
    Ok(())
}

// a free-surface plate of body b: given by its mesh, or generated over a loop of a trimmed waterline
// (its own body's, or the one its item body names); a plate of a trimmed body is generated
fn check_plate(b: Obj, p: Obj, trimmed: bool) -> Result<()> {
    let at = p.prefix();
    if !generated_plate(p) {
        if p.has("body") {
            return Err(Error::value(format!(
                "{at}body names the body whose waterline loop a generated plate covers: give the point inside as well"
            )));
        }
        if trimmed {
            return Err(Error::value(format!(
                "{at}a free-surface plate of a trimmed body is generated from its waterline: give a point of the loop \
                 (inside), not a mesh (the plate's elements lie in z = 0 of the untrimmed hull, and any angle tilts \
                 them out of it)"
            )));
        }
        if !p.has("mesh") {
            return Err(Error::value(format!(
                "{at}a FreeSurfacePlate needs its mesh, or a point inside a loop of a trimmed waterline (inside)"
            )));
        }
        return Ok(());
    }
    if p.has("mesh") || p.has("ranges") {
        return Err(Error::value(format!(
            "{at}a generated plate (inside) takes no mesh and no ranges: it is made from the waterline"
        )));
    }
    let name = waterline_body(b, p);
    if name == b.name() {
        if !trimmed {
            return Err(Error::value(format!(
                "{at}a generated plate covers a loop of a trimmed waterline: the body has no trim (give the plate's \
                 mesh, or name the trimmed body whose loop it covers)"
            )));
        }
        return Ok(());
    }
    if trimmed {
        return Err(Error::value(format!(
            "{at}a generated plate of a trimmed body covers a loop of its own waterline (leave body unset)"
        )));
    }
    let Some(o) = b.document().root().children("bodies").into_iter().find(|x| x.name() == name) else {
        return Err(Error::value(format!("{at}body: no body {}", py_str_repr(name))));
    };
    if !o.has("trim") || !o.has("mesh") {
        return Err(Error::value(format!(
            "{at}body: the body {} has no trimmed hull (a trim and a mesh): a generated plate covers a loop of a \
             trimmed waterline",
            py_str_repr(name)
        )));
    }
    Ok(())
}

/// A coupling in the messages: its path and the body it names.
fn coupling_text(c: Obj) -> String {
    match c.body("body") {
        Some(other) => format!("{} (with {})", c.path(), py_str_repr(other)),
        None => c.path(),
    }
}

// the couplings of a body: each names another body, once, with a stiffness or a damping; the
// columns of a matrix are about the origin of the other body, which must be usable
fn check_couplings(b: Obj) -> Result<()> {
    let mut seen: Vec<&str> = Vec::new();
    for c in b.children("couplings") {
        let at = coupling_text(c);
        let Some(other) = c.body("body") else {
            return Err(Error::value(format!("{at}: a coupling needs its body (the other body, by name)")));
        };
        if other == b.name() {
            return Err(Error::value(format!(
                "{at}: a coupling names another body (the body's own block is additional_stiffness, additional_damping)"
            )));
        }
        if seen.contains(&other) {
            return Err(Error::value(format!(
                "{at}: a second coupling with the body {} (one coupling per other body)",
                py_str_repr(other)
            )));
        }
        seen.push(other);
        if !c.has("stiffness") && !c.has("damping") {
            return Err(Error::value(format!("{at}: a coupling needs a stiffness or a damping")));
        }
        let Some(o) = b.document().root().child("bodies", other) else {
            return Err(Error::value(format!("{at}: body: unknown body {}", py_str_repr(other))));
        };
        for (m, origin) in [("stiffness", "stiffness_origin"), ("damping", "damping_origin")] {
            if c.has(m) && o.origin(origin)? == OriginItem::Named("center_of_gravity") && !o.has("center_of_gravity") {
                return Err(Error::value(format!(
                    "{at}: the columns of its {m} are the motions of {} about its {origin}='center_of_gravity', which it lacks (give that body a center_of_gravity)",
                    o.path()
                )));
            }
        }
    }
    Ok(())
}

/// The points of an array item (N, 3), none when it is unset.
fn points_of(o: Obj, item: &str) -> Vec<Vec3> {
    let v = o.optional_reals(item).unwrap_or_default();
    v.chunks_exact(3).map(|p| v3(p[0], p[1], p[2])).collect()
}

/// The field points in the sea (first_order.field_points), global.
pub fn sea_points(doc: &Document) -> Vec<Vec3> {
    points_of(single(doc.root(), "first_order"), "field_points")
}

/// The points in the tanks as the tanks give them (body axes), the tanks of the bodies in order,
/// and per point its tank (k >= 1 the k-th tank of the bodies in order, as `field_points`).
pub fn tank_points(doc: &Document) -> (Vec<Vec3>, Vec<i32>) {
    let (mut pts, mut tanks) = (Vec::new(), Vec::new());
    let mut k = 0;
    for b in doc.root().children("bodies") {
        for t in b.children("tanks") {
            k += 1;
            for p in points_of(t, "field_points") {
                pts.push(p);
                tanks.push(k);
            }
        }
    }
    (pts, tanks)
}

/// The field points of the problem: the points in the sea (first_order.field_points), then the
/// points of every tank of the bodies in order, placed with their body (body axes to global); per
/// point its tank as the problem numbers them (0 the sea, k >= 1 the k-th tank of the bodies in
/// order). The solve computes its field outputs at these points; the results give those of the sea
/// and those of the tanks apart (`model_solve::split_field_points`).
pub fn field_points(doc: &Document) -> Result<(Vec<Vec3>, Vec<i32>)> {
    let root = doc.root();
    let mut pts = points_of(single(root, "first_order"), "field_points");
    let mut tanks = vec![0; pts.len()];
    let mut k = 0;
    for b in root.children("bodies") {
        let place = Placement { position: point(b, "position")?, yaw_deg: b.real("yaw")? };
        for t in b.children("tanks") {
            k += 1;
            for p in points_of(t, "field_points") {
                pts.push(place.apply(p));
                tanks.push(k);
            }
        }
    }
    Ok((pts, tanks))
}

// whether a switch of the items is on (the switches read in order up to the first one on)
fn any(items: &[&str], on: impl Fn(&str) -> Result<bool>) -> Result<bool> {
    for item in items {
        if on(item)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn omegas_of(f: Obj) -> Result<Vec<f64>> {
    if f.choice("frequency_input")? == "periods" {
        let mut om = Vec::new();
        for &t in f.reals("periods")? {
            om.push(if t != 0.0 { 2.0 * std::f64::consts::PI / t } else { f64::INFINITY });
        }
        return Ok(om);
    }
    Ok(f.reals("omegas")?.to_vec())
}

/// The rules between the items of an object and between objects; ErrorKind::Value naming the
/// object.
pub fn check_objects(doc: &Document) -> Result<()> {
    let root = doc.root();
    let out = single(root, "first_order");
    let second = single(root, "second_order");
    if root.children("bodies").is_empty() {
        return Err(Error::value("the model has no bodies"));
    }
    let f = single(root, "environment");
    let input = f.choice("frequency_input")?;
    let om = omegas_of(f)?;
    if om.is_empty() && (out.boolean("radiation")? || out.boolean("diffraction")?) {
        return Err(Error::value(format!(
            "environment: {input} is empty (allowed only with first_order.radiation = false and diffraction = false)"
        )));
    }
    // the second order: Newman's approximation from a drift switched on; a QTF at some frequencies
    let drifts = ["drift_pressure_integration", "drift_control_surface", "drift_momentum_conservation"];
    let on = |item: &str| second.boolean(item);
    if on("qtf_newman")? && !any(&drifts, on)? {
        return Err(Error::value(
            "second_order: qtf_newman makes Newman's approximation from the drifts switched on: switch a drift on (drift_pressure_integration, drift_control_surface or drift_momentum_conservation)",
        ));
    }
    let qtfs = [
        "qtf_quadratic_pressure_integration",
        "qtf_quadratic_control_surface",
        "qtf_potential_direct",
        "qtf_potential_indirect",
        "qtf_newman",
    ];
    if any(&qtfs, on)? && !on("qtf_difference")? && !on("qtf_sum")? {
        return Err(Error::value(
            "second_order: a QTF is switched on but qtf_difference and qtf_sum are both false: switch one of them on",
        ));
    }
    check_switched(second)?;
    check_body_objects(doc, Some(second))
}

/// The rules between the items of the bodies, the lids and the fixed dampers and between them (the
/// rules of `check_objects` that the queries of a body apply); with `second` also the rule of a
/// body's control-surface waterline term with the second order. ErrorKind::Value naming the object.
pub fn check_body_objects(doc: &Document, second: Option<Obj>) -> Result<()> {
    let root = doc.root();
    if root.children("bodies").is_empty() {
        return Err(Error::value("the model has no bodies"));
    }
    for b in root.children("bodies") {
        check_body(b)?;
        let Some(second) = second else { continue };
        if control_method(b)? != "none"
            && b.choice("control_surface_waterline_term")? == "body"
            && !second.boolean("drift_control_surface")?
        {
            return Err(Error::value(format!(
                "{}control_surface_waterline_term = 'body' needs second_order.drift_control_surface",
                b.prefix()
            )));
        }
    }
    for (li, l) in root.children("lids").iter().enumerate() {
        let here = Location { lid: Some(li), ..Location::default() };
        need_mesh(*l, "DampingLid").located(here)?;
        let Some(w) = l.body("placed_with") else {
            continue;
        };
        let Some(b) = root.child("bodies", w) else {
            return Err(
                Error::value(format!("{}placed_with: the unknown body {}", l.prefix(), py_str_repr(w))).at(here)
            );
        };
        let here = Location { body: Some(b.index()), ..here };
        let lm = MeshRef::of(*l).expect("checked above");
        if MeshRef::of(b).is_none_or(|r| r.name != lm.name) {
            return Err(Error::value(format!(
                "{}a lid placed with the body '{}' takes elements of that body's mesh (or give it in global coordinates without placed_with)",
                l.prefix(),
                b.name()
            ))
            .at(here));
        }
        if b.has("trim") {
            return Err(Error::value(format!(
                "{}a lid from the elements of the trimmed body '{}' is not available (give the lid as a mesh of its own)",
                l.prefix(),
                b.name()
            ))
            .at(here));
        }
    }
    for d in root.children("fixed_dampers") {
        need_mesh(d, "FixedDamper")?;
    }
    Ok(())
}

// ------------------------------------------------------------------ the problem

struct Builder<'a> {
    doc: &'a Document,
    m: &'a mut Problem,
    meshes: Vec<(String, i32)>, // the meshes added as they are, by name, in order of first use
    lids: Vec<(i32, Vec<Range>)>, // per lid: the body that places it and its elements of the body's engine mesh
}

impl<'a> Builder<'a> {
    fn run(&mut self) -> Result<()> {
        let root = self.doc.root();
        let env = single(root, "environment");
        self.m.env.depth = env.real("depth")?;
        self.m.env.gravity = env.real("gravity")?;
        self.m.env.density = env.real("density")?;
        self.options(single(root, "options"), single(root, "first_order"), single(root, "second_order"), env)?;
        let lids = root.children("lids");
        self.lids = vec![(-1, Vec::new()); lids.len()];
        let placed = placed_fixed_dampers(self.doc)?;
        let mut attached = vec![false; root.children("fixed_dampers").len()];
        for (bi, b) in root.children("bodies").iter().enumerate() {
            for &k in &placed[bi] {
                attached[k] = true;
            }
            self.body(*b, bi, &placed[bi])?;
        }
        for (li, l) in lids.iter().enumerate() {
            let mut d = Lid { name: l.name().to_string(), ..Lid::default() };
            if l.body("placed_with").is_some() {
                (d.body, d.panels) = self.lids[li].clone();
            } else {
                let place = Location { lid: Some(li), ..Location::default() };
                let r = MeshRef::of(*l).expect("a lid has its mesh");
                d.mesh = some_index(self.mesh_index(&r, &l.prefix(), place).located(place)?);
            }
            d.damping_factor = l.real("damping_factor")?;
            self.m.lids.push(d);
        }
        {
            // the potential QTF's free surface: its zones (the panelled zone from a mesh or generated)
            let fs = single(root, "second_order");
            let method = fs.choice("free_surface_method")?;
            let given = method == "mesh";
            let d = &mut self.m.opt;
            // a number or "auto" (None)
            let real_or_auto = |item: &str| fs.real_or_name(item);
            let int_or_auto = |item: &str| -> Result<Option<i32>> {
                fs.integer_or_name(item)?.map(|v| int32_of(v, &format!("second_order.{item}"))).transpose()
            };
            // the items of the generated zone, or of the given mesh (the others kept unused)
            d.potential_qtf_radius = if given { None } else { real_or_auto("free_surface_inner_radius")? };
            d.potential_qtf_panel_size = if given { None } else { real_or_auto("free_surface_panel_size")? };
            d.potential_qtf_panels = method != "cells";
            d.free_surface_points = if given {
                int32_of(fs.integer("free_surface_quadrature_points")?, "second_order.free_surface_quadrature_points")?
            } else {
                1
            };
            d.potential_qtf_annuli = int_or_auto("quadrature_annuli")?.unwrap_or(-1).max(0);
            d.potential_qtf_annulus_step = real_or_auto("quadrature_radius_step")?;
            d.potential_qtf_radial_nodes = int_or_auto("quadrature_radial_nodes")?.unwrap_or(-1);
            if d.potential_qtf_radial_nodes < 0 {
                d.potential_qtf_radial_nodes = 12;
            }
            d.potential_qtf_azimuthal_nodes = int_or_auto("quadrature_azimuthal_nodes")?;
            d.potential_qtf_segments = int_or_auto("outer_circle_segments")?;
            d.potential_qtf_order = int_or_auto("expansion_order")?;
            if let Some(r) = MeshRef::given(fs, FREE_SURFACE)? {
                self.m.opt.free_surface_mesh = some_index(self.mesh_index(
                    &r,
                    &format!("{}free_surface_mesh: ", fs.prefix()),
                    Location::default(),
                )?);
            }
        }
        for (k, d) in root.children("fixed_dampers").iter().enumerate() {
            if attached[k] {
                continue;
            }
            let r = MeshRef::of(*d).expect("a damper has its mesh");
            let mesh = self.mesh_index(&r, &d.prefix(), Location::default())?;
            self.m.fixed_dampers.push(FixedDamper {
                mesh,
                epsilon: d.real("damping_factor")?,
                index: k as i32,
                name: d.name().to_string(),
                place: d.position() as i32,
            });
        }
        // auto: the total formulation with lids (the incident potential does not satisfy the lid
        // condition, so the scattered one has a boundary value on them) and with free-surface
        // plates (the water under a plate is shielded from the incident wave, as in a deep
        // column: the scattered potential there is nearly -phi_I and its discretization error,
        // proportional to it, swamps the small total; the total potential is small there and so
        // is its error; theory §9.7), the scattered one otherwise and with low-order thin sheets,
        // which need it (rule 8)
        let formulation = single(root, "first_order").choice("diffraction_formulation")?;
        let lo_sheets = !self.m.opt.higher_order
            && (!self.m.fixed_dampers.is_empty() || self.m.bodies.iter().any(|b| b.has_sheets()));
        let plates = self.m.bodies.iter().any(|b| !b.plates.is_empty());
        self.m.opt.formulation = match formulation {
            "total" => Formulation::Total,
            "scattered" => Formulation::Scattered,
            _ if (!self.m.lids.is_empty() || plates) && !lo_sheets => Formulation::Total,
            _ => Formulation::Scattered,
        };
        let f = single(root, "environment");
        self.m.omegas = omegas_of(f)?;
        self.m.headings = f.reals("headings")?.to_vec();
        // the field points: those of the sea, then those of the tanks (0 the sea, k >= 1 the k-th
        // tank of the bodies in order); the tags only with a point in a tank
        let (pts, tanks) = field_points(self.doc)?;
        if !pts.is_empty() {
            self.m.has_field_points = true;
            if tanks.iter().any(|&k| k != 0) {
                self.m.field_tanks = tanks;
            }
            self.m.field_points = pts;
        }
        // the higher order: the surfaces of other meshes as elements of the body meshes
        merge_other_meshes(self.m)?;
        self.m.invalidate();
        Ok(())
    }

    // a mesh added to the problem with the stored elements of its elements (None: its own), its index
    fn add_mesh(&mut self, d: MeshDesc, origin: Option<MeshOrigin>) -> Result<i32> {
        let index = self.m.meshes.len();
        let mesh = engine_mesh(d, index)?;
        let origin = origin.unwrap_or_else(|| MeshOrigin {
            stored: mesh.count,
            records: (1..=mesh.count).collect(),
            ..MeshOrigin::default()
        });
        self.m.meshes.push(mesh);
        self.m.origins.push(origin);
        Ok(index as i32)
    }

    // the mesh of a reference as a mesh of the problem: a whole mesh once (by name), the elements of
    // ranges as a mesh of their own (at: the prefix of the object that gives it, place its location)
    fn mesh_index(&mut self, r: &MeshRef, at: &str, place: Location) -> Result<i32> {
        if r.ranges.is_empty() {
            if let Some((_, i)) = self.meshes.iter().find(|(n, _)| *n == r.name) {
                return Ok(*i);
            }
        }
        let mut d = desc_of(self.doc, &r.name).located(place)?;
        let mut origin = None;
        if !r.ranges.is_empty() {
            let n = stored_records(&d);
            for x in &r.ranges {
                if x.last > n {
                    return Err(Error::value(format!(
                        "{at}the element range ({}, {}) is beyond the mesh {} ({n} elements)",
                        x.first,
                        x.last,
                        py_str_repr(&r.name)
                    ))
                    .at(place));
                }
            }
            let mut keep = vec![false; n.max(0) as usize];
            for i in r.records(n) {
                keep[(i - 1) as usize] = true;
            }
            keep_records(&mut d, &keep);
            let records = (1..=n).filter(|&i| keep[(i - 1) as usize]).collect();
            origin = Some(MeshOrigin { stored: n, records, ..MeshOrigin::default() });
        }
        let i = self.add_mesh(d, origin)?;
        if r.ranges.is_empty() {
            self.meshes.push((r.name.clone(), i));
        }
        Ok(i)
    }

    fn options(&mut self, o: Obj, out: Obj, second: Obj, env: Obj) -> Result<()> {
        let d = &mut self.m.opt;
        let on = |item: &str| second.boolean(item);
        d.diffraction = out.boolean("diffraction")?;
        d.radiation = out.boolean("radiation")?;
        d.rao = out.boolean("motion")?;
        d.logarithmic = o.boolean("logarithmic_integration")?;
        d.domain = choice(&DOMAINS, env.choice("domain")?);
        d.channel_width = if env.has("channel_width") { env.real("channel_width")? } else { f64::NAN };
        d.channel_reflect = env.real("channel_reflect")?;
        d.higher_order = o.boolean("higher_order")?;
        // the drifts by their methods and heading pairs (the Kochin angles of momentum
        // conservation given, or "auto", chosen by the engine: -1)
        let angles = second.integer_or_name("drift_momentum_conservation_angles")?;
        let momentum = on("drift_momentum_conservation")?;
        d.drift_angles = match angles {
            _ if !momentum => 0,
            None => -1,
            Some(n) => int32_of(n, "second_order.drift_momentum_conservation_angles")?,
        };
        d.pressure_drift = on("drift_pressure_integration")?;
        d.control_surface_drift = on("drift_control_surface")?;
        d.drift_pairs = on("drift_bidirectional")?;
        // the QTFs (theory §12.2): the headings once, the parts by their methods; the engine
        // computes what the quadratic part rests on (the pressure drift; the pressure QTFs for the
        // control surface's hull values) and returns it only when asked for
        let qtf_headings = if on("qtf_bidirectional")? { QtfHeadings::Pairs } else { QtfHeadings::Equal };
        d.qtf_pairs = qtf_headings == QtfHeadings::Pairs;
        let pressure = on("qtf_quadratic_pressure_integration")?;
        let cs = on("qtf_quadratic_control_surface")?;
        d.qtf_frequencies = match (on("qtf_difference")?, on("qtf_sum")?) {
            (true, false) => QtfFrequencies::Difference,
            (false, true) => QtfFrequencies::Sum,
            _ => QtfFrequencies::Both,
        };
        // the quadratic QTFs are computed at both (the potential ones at those asked for alone)
        match d.qtf_frequencies {
            QtfFrequencies::Difference => d.hidden.extend(QTF_SUMS),
            QtfFrequencies::Sum => d.hidden.extend(QTF_DIFFERENCES),
            QtfFrequencies::Both => {}
        }
        // the bidirectional pressure and control-surface drifts: the diagonal of the quadratic QTFs
        // of every heading pair
        let drift_qtf = d.drift_pairs && (d.pressure_drift || d.control_surface_drift);
        if pressure || cs || drift_qtf {
            d.pressure_qtf = if pressure || cs { qtf_headings } else { QtfHeadings::Pairs };
            if drift_qtf {
                d.pressure_qtf = QtfHeadings::Pairs;
            }
            d.qtf_diagonal = !(pressure || cs);
            if !d.pressure_drift {
                d.hidden.extend([QuantityId::DriftPressureIntegration, QuantityId::DriftPressureIntegrationBodyAxes]);
            }
            if !pressure {
                d.hidden.extend([
                    QuantityId::QtfQuadraticPressureIntegrationDifference,
                    QuantityId::QtfQuadraticPressureIntegrationSum,
                ]);
            }
            if !cs {
                d.hidden.extend([
                    QuantityId::QtfQuadraticControlSurfaceDifference,
                    QuantityId::QtfQuadraticControlSurfaceSum,
                ]);
            }
            if cs || (drift_qtf && d.control_surface_drift) {
                d.control_surface_qtf = true;
                if !d.control_surface_drift {
                    d.hidden.extend([QuantityId::DriftControlSurface, QuantityId::DriftControlSurfaceBodyAxes]);
                }
                d.control_surface_drift = true;
            }
            d.pressure_drift = true;
        }
        // Newman's approximation from every drift switched on (no drift is computed for it alone)
        let newman = on("qtf_newman")?;
        d.newman = NewmanDrifts {
            pressure_integration: newman && on("drift_pressure_integration")?,
            control_surface: newman && on("drift_control_surface")?,
            momentum_conservation: newman && momentum,
        };
        d.newman_geometric = second.choice("qtf_newman_average")? != "arithmetic";
        let direct = on("qtf_potential_direct")?;
        let indirect = on("qtf_potential_indirect")?;
        if direct || indirect {
            d.potential_qtf = qtf_headings;
            d.potential_qtf_direct = direct && !indirect;
            d.potential_qtf_both = direct && indirect;
        }
        d.field_velocity = out.boolean("field_velocity")?;
        d.field_pressure = out.boolean("field_pressure")?;
        d.log_tolerance = o.real("log_tolerance")?;
        d.body_pressure = out.boolean("body_pressure")?;
        d.body_velocity = out.boolean("body_velocity")?;
        d.frequency_threads = int32_of(o.integer("frequency_threads")?, "options.frequency_threads")?;
        d.inner_threads = int32_of(o.integer("inner_threads")?, "options.inner_threads")?;
        d.memory_check = o.boolean("memory_check")?;
        Ok(())
    }

    // the origin of an origin item of a body and its point
    fn origin(o: Obj, item: &str) -> Result<(Origin, Vec3)> {
        Ok(match o.origin(item)? {
            OriginItem::Named(n) => {
                let code = if n == "center_of_gravity" { Origin::CenterOfGravity } else { Origin::RotationCenter };
                (code, v3(0.0, 0.0, 0.0))
            }
            OriginItem::Point(p) => (Origin::Point, v3(p[0], p[1], p[2])),
        })
    }

    // the generated control surface of a body: a mesh as the engine made it as a mesh object
    fn control_surface(&mut self, c: Obj, b: &Body, at: &str, covers: Vec<Vec<[f64; 2]>>) -> Result<i32> {
        let method = control_method(c)?;
        let item = |n: &str| format!("control_surface_{n}");
        let real = |n: &str| c.real(&item(n));
        let reals = |n: &str| c.reals(&item(n));
        let pairs = |v: &[f64]| -> Vec<[f64; 2]> { v.chunks_exact(2).map(|p| [p[0], p[1]]).collect() };
        let mut s = ControlSurfaceInput {
            kind: ControlSurfaceKind::Automatic,
            values: Vec::new(),
            panel_size: real("panel_size")?,
            radius: -1.0,
            depth: 0.0,
            free_surface: true,
            polygon: Vec::new(),
            partitions: Vec::new(),
            label: String::new(),
        };
        let g = if method == "automatic" {
            s.label = "automatic".to_string();
            s.depth = real("depth")?;
            // the outer surface of outer; the other item kept unused
            if c.choice(&item("outer"))? == "circle" {
                s.radius = real("radius")?;
            } else {
                s.polygon = pairs(reals("boundary")?);
            }
            // the polylines of partitions, each ended by a row of NaN
            let mut line: Vec<[f64; 2]> = Vec::new();
            for p in pairs(c.optional_reals(&item("partitions")).unwrap_or_default()) {
                if p[0].is_nan() || p[1].is_nan() {
                    if !line.is_empty() {
                        s.partitions.push(std::mem::take(&mut line));
                    }
                } else {
                    line.push(p);
                }
            }
            if !line.is_empty() {
                s.partitions.push(line);
            }
            let mesh = &self.m.meshes[b.mesh as usize];
            control_surface_gdf_covered(&s, Some((mesh, 0.0 - b.position[2])), covers) // +0.0 at z = 0
        } else if method == "cylinder" {
            s.kind = ControlSurfaceKind::Cylinder;
            s.label = "cylinder control surface".to_string();
            s.values = vec![real("radius")?, real("depth")?, real("inner_radius")?];
            control_surface_gdf(&s, None)
        } else {
            s.kind = ControlSurfaceKind::Ellipsoid;
            s.label = "ellipsoid control surface".to_string();
            s.values = reals("semi_axes")?.to_vec();
            s.values.extend(reals("inner_semi_axes")?);
            control_surface_gdf(&s, None)
        };
        let g = g.map_err(|e| {
            let m = format!("{at}control_surface: {}", e.message());
            e.with_message(m)
        })?;
        self.add_mesh(gdf_desc(&g)?, None)
    }

    // the mesh of plate s of body o as a mesh of the problem: its mesh of its own (r), or the mesh
    // generated over another body's waterline loop; -1 for the elements of the body's mesh (given,
    // generated by the trim, or the first plate of a body of generated plates only)
    fn plate_mesh(&mut self, o: Obj, s: usize, bm: &BodyMesh, r: &Option<MeshRef>, here: Location) -> Result<i32> {
        let fp = o.children("free_surface_plates")[s];
        if generated_plate(fp) && waterline_body(o, fp) != o.name() && !(bm.own.is_empty() && s == 0) {
            let d = other_plate_desc(o, s).located(here)?;
            return self.add_mesh(d, None);
        }
        match r {
            Some(r) => self.mesh_index(r, &fp.prefix(), here),
            None => Ok(-1),
        }
    }

    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn body(&mut self, o: Obj, bi: usize, fixed_dampers: &[usize]) -> Result<()> {
        let at = o.prefix();
        let bm = body_mesh(o, fixed_dampers)?;
        let own = bm.own.clone();
        let other = |r: &Option<MeshRef>| r.as_ref().filter(|r| r.name != own).cloned();
        // the meshes first, in the order of their first use: the body's, the interior surface, the
        // dipoles, the control surface and the pressure surfaces given as meshes of their own
        let mesh = if !own.is_empty() && !bm.trimmed && bm.source.len() as i32 == bm.stored {
            self.mesh_index(&MeshRef { name: own.clone(), ranges: Vec::new() }, "", Location::default())?
        } else {
            // (a body of generated plates only: their mesh)
            let origin = MeshOrigin {
                stored: bm.stored,
                records: bm.source.clone(),
                copy: bm.copy.clone(),
                part: bm.part.clone(),
                joined: Vec::new(),
                plate: if bm.trimmed { bm.plate.clone() } else { Vec::new() },
            };
            self.add_mesh(bm.desc.clone(), Some(origin))?
        };
        let here = at_body(self.m.bodies.len() as i32); // its index
        other_roles(o, &own).located(here)?;
        let interior_ref = MeshRef::given(o, INTERIOR_SURFACE)?;
        let interior_mesh = match other(&interior_ref) {
            Some(r) => self.mesh_index(&r, &format!("{at}interior_surface_mesh: "), here)?,
            None => -1,
        };
        // the dipoles on other meshes: the elements of one other mesh (none twice: other_roles)
        let mut dipole_other: Option<MeshRef> = None;
        for d in o.children("dipoles") {
            let Some(r) = other(&MeshRef::of(d)) else { continue };
            match &mut dipole_other {
                None => dipole_other = Some(r),
                Some(x) if x.name == r.name => {
                    x.ranges.extend(r.ranges);
                    x.ranges.sort_by_key(|q| q.first);
                }
                Some(x) => {
                    return Err(Error::value(format!(
                        "{at}dipoles: the dipoles take elements of the body's mesh and of at most one other mesh, not of {} and {}",
                        py_str_repr(&x.name),
                        py_str_repr(&r.name)
                    )));
                }
            }
        }
        let dipole_mesh = match &dipole_other {
            Some(r) => self.mesh_index(r, &format!("{at}dipoles: "), here)?,
            None => -1,
        };
        let mut b = Body { name: o.name().to_string(), mesh, ..Body::default() };
        b.position = point(o, "position")?;
        b.yaw = o.real("yaw")?;
        let cs = control_method(o)?;
        let plates = o.children("free_surface_plates");
        // the plates' meshes of their own: after the pressure surfaces', or before an automatic
        // control surface (it leaves the waterline under the plates out)
        let early = cs == "automatic" && !plates.is_empty();
        let mut plate_meshes: Vec<i32> = Vec::new();
        if early {
            for (s, fp) in plates.iter().enumerate() {
                let m = self.plate_mesh(o, s, &bm, &other(&MeshRef::of(*fp)), here)?;
                plate_meshes.push(m);
            }
        }
        let control_surface = match cs {
            "none" => -1,
            "mesh" => self.mesh_index(
                &MeshRef::given(o, CONTROL_SURFACE)?.expect("checked"),
                &format!("{at}control_surface_mesh: "),
                here,
            )?,
            _ => {
                // the elements of the plates as polygons in body axes
                let mut covers: Vec<Vec<[f64; 2]>> = Vec::new();
                for (s, &pm) in plate_meshes.iter().enumerate() {
                    let (mi, recs): (usize, Vec<usize>) = if pm >= 0 {
                        (pm as usize, (0..self.m.meshes[pm as usize].count as usize).collect())
                    } else {
                        let rs = bm.ranges(Part::Plate(s));
                        (mesh as usize, rs.iter().flat_map(|x| (x.first - 1) as usize..x.last as usize).collect())
                    };
                    for rec in recs {
                        covers.extend(crate::plate_geometry::element_outlines(&self.m.meshes[mi], rec));
                    }
                }
                self.control_surface(o, &b, &at, covers)?
            }
        };
        let mut pressure_meshes: Vec<i32> = Vec::new();
        for ps in o.children("pressure_surfaces") {
            pressure_meshes.push(match other(&MeshRef::of(ps)) {
                Some(r) => self.mesh_index(&r, &ps.prefix(), here)?,
                None => -1,
            });
        }
        if !early {
            for (s, fp) in plates.iter().enumerate() {
                let m = self.plate_mesh(o, s, &bm, &other(&MeshRef::of(*fp)), here)?;
                plate_meshes.push(m);
            }
        }

        b.rotation_center = point(o, "rotation_center")?;
        for m in o.texts("modes")? {
            b.modes.push(index_of(&MODE_NAMES, m) + 1);
        }
        if o.has("center_of_gravity") {
            b.has_cg = true;
            b.cg = point(o, "center_of_gravity")?;
        }
        b.mass = match o.has("mass") {
            false => None,
            true => Some(o.real_or_name("mass")?.map_or(Mass::Displaced, Mass::Given)),
        };
        // the inertia of inertia_input (every item has a value; the other side is not read)
        let components = o.choice("inertia_input")? == "components";
        b.has_inertia = true;
        if components {
            for k in 0..6 {
                b.inertia[k] = o.real(INERTIA[k])?;
            }
        }
        if let Some(bp) = o.optional_reals("body_points") {
            for i in 0..o.shape("body_points")?[0] as usize {
                b.body_points.push(v3(bp[3 * i], bp[3 * i + 1], bp[3 * i + 2]));
            }
        }
        b.body_point_neighbors = int32_of(o.integer("body_point_neighbors")?, &format!("{at}body_point_neighbors"))?;
        if !components {
            let gy = o.reals("radii_of_gyration")?;
            b.has_gyration = true;
            if o.shape("radii_of_gyration")?.len() == 1 {
                for k in 0..3 {
                    b.gyration[4 * k] = gy[k]; // three values: the diagonal
                }
            } else {
                b.gyration[..gy.len()].copy_from_slice(gy);
            }
        }
        (b.inertia_origin, b.inertia_point) = Builder::origin(o, "inertia_origin")?;
        (b.stiffness_origin, b.stiffness_point) = Builder::origin(o, "stiffness_origin")?;
        (b.damping_origin, b.damping_point) = Builder::origin(o, "damping_origin")?;
        let ks = o.optional_reals("additional_stiffness");
        let kd = o.optional_reals("additional_damping");
        if let Some(ks) = ks {
            let shape = o.shape("additional_stiffness")?;
            b.stiffness = ks.to_vec();
            b.stiffness_rows = shape[0] as i32;
            b.stiffness_cols = shape[1] as i32;
            if b.stiffness.is_empty() {
                return Err(Error::value(format!("body '{}': additional_stiffness is empty", b.name)).at(here));
            }
        }
        if let Some(kd) = kd {
            let shape = o.shape("additional_damping")?;
            b.damping = kd.to_vec();
            b.damping_rows = shape[0] as i32;
            b.damping_cols = shape[1] as i32;
            if b.damping.is_empty() {
                return Err(Error::value(format!("body '{}': additional_damping is empty", b.name)).at(here));
            }
        }
        // the couplings with the other bodies: the body's six rows of all 6N rigid coordinates
        // (6 x 6N), its own block where the body is, the block of a coupling where its body is
        let bodies = self.doc.root().children("bodies");
        for (m, own, cols, rows, mat) in [
            ("stiffness", &ks, &mut b.stiffness_cols, &mut b.stiffness_rows, &mut b.stiffness),
            ("damping", &kd, &mut b.damping_cols, &mut b.damping_rows, &mut b.damping),
        ] {
            let blocks: Vec<(usize, &[f64])> = o
                .children("couplings")
                .iter()
                .filter_map(|c| {
                    let a = c.optional_reals(m)?;
                    let j = bodies.iter().position(|x| Some(x.name()) == c.body("body"))?;
                    Some((j, a))
                })
                .collect();
            if blocks.is_empty() {
                continue;
            }
            let n6 = 6 * bodies.len();
            let mut k = vec![0.0; 6 * n6];
            let mut put = |j: usize, a: &[f64]| {
                for r in 0..6 {
                    k[r * n6 + 6 * j..r * n6 + 6 * j + 6].copy_from_slice(&a[6 * r..6 * r + 6]);
                }
            };
            if let Some(own) = own {
                put(bi, own);
            }
            for (j, a) in &blocks {
                put(*j, a);
            }
            *mat = k;
            *rows = 6;
            *cols = n6 as i32;
        }
        b.excitation =
            if o.choice("excitation_method")? == "diffraction" { Excitation::Pressure } else { Excitation::Haskind };
        // the interior surface: elements of the body's mesh, a mesh of its own, generated by the solve,
        // or (higher order, trimmed) the elements the trim generated
        // (detect: the elements found in z = 0, none without them)
        let m = interior_method(o)?;
        if m != "none" {
            if m == "detect" {
                if bm.parts.iter().any(|(p, _)| *p == Part::Interior) {
                    b.interior_method = InteriorMethod::PanelRange;
                    b.interior_panels = bm.ranges(Part::Interior);
                }
            } else if m != "mesh" {
                b.interior_method = choice(&INTERIOR_METHODS, m);
                if bm.trimmed && self.m.opt.higher_order && m == "waterline" {
                    b.interior_method = InteriorMethod::PanelRange;
                    b.interior_panels = bm.generated();
                }
            } else if interior_mesh >= 0 {
                b.interior_method = InteriorMethod::ExternalMesh;
                b.interior_mesh = Some(interior_mesh as usize);
            } else {
                b.interior_method = InteriorMethod::PanelRange;
                b.interior_panels = bm.ranges(Part::Interior);
            }
        }
        // the dipoles of the body and those inside its tanks (the tank's elements, its fluid)
        b.dipole_panels = Vec::new();
        for k in 0..o.children("dipoles").len() {
            b.dipole_panels.extend(bm.ranges(Part::Dipole(k)));
        }
        let tanks = o.children("tanks");
        for (t, x) in tanks.iter().enumerate() {
            for k in 0..x.children("dipoles").len() {
                b.dipole_panels.extend(bm.ranges(Part::TankDipole(t, k)));
            }
        }
        b.dipole_panels.sort_by_key(|r| r.first);
        b.dipole_mesh = some_index(dipole_mesh);
        // the owners of the dipoles for the queries: on the body's mesh their elements, on another
        // mesh per element of that mesh the dipole whose ranges hold its stored element
        for k in 0..o.children("dipoles").len() {
            b.dipole_owners.push(DipoleOwner { panels: bm.ranges(Part::Dipole(k)), index: k as i32 });
        }
        for x in tanks.iter() {
            for k in 0..x.children("dipoles").len() {
                let t = tanks.iter().position(|y| y.id() == x.id()).expect("a tank of the body");
                b.dipole_owners.push(DipoleOwner { panels: bm.ranges(Part::TankDipole(t, k)), index: k as i32 });
            }
        }
        if dipole_mesh >= 0 {
            let n = self.m.stored_records(dipole_mesh);
            let refs: Vec<Option<MeshRef>> = o.children("dipoles").iter().map(|d| other(&MeshRef::of(*d))).collect();
            b.dipole_mesh_owner = (0..self.m.meshes[dipole_mesh as usize].count)
                .map(|rec| {
                    let s = self.m.stored_record(dipole_mesh, rec);
                    let holds = |r: &MeshRef| r.records(n).contains(&s);
                    refs.iter().position(|r| r.as_ref().is_some_and(holds)).map_or(-1, |k| k as i32)
                })
                .collect();
        }
        b.control_surface = (control_surface >= 0).then_some(control_surface as usize);
        if cs != "none" {
            b.control_surface_alternative = if o.choice("control_surface_waterline_term")? == "body" {
                WaterlineTerm::Body
            } else {
                WaterlineTerm::ControlSurface
            };
        }
        for (k, x) in o.children("dipole_dampers").iter().enumerate() {
            b.dampers.push(Damper {
                panels: bm.ranges(Part::Damper(k)),
                epsilon: x.real("damping_factor")?,
                condition: DamperCondition::FixedPorous,
                index: k as i32,
                name: x.name().to_string(),
                label: Part::Damper(k).text(o),
            });
        }
        // the dampers inside the tanks, moving with the body (after those of the body)
        for (t, x) in tanks.iter().enumerate() {
            for (k, y) in x.children("dipole_dampers").iter().enumerate() {
                b.dampers.push(Damper {
                    panels: bm.ranges(Part::TankDamper(t, k)),
                    epsilon: y.real("damping_factor")?,
                    condition: DamperCondition::FixedPorous,
                    index: k as i32, // its place in the dampers of its tank
                    name: y.name().to_string(),
                    label: Part::TankDamper(t, k).text(o),
                });
            }
        }
        let water = self.doc.root().children("fixed_dampers");
        for &k in fixed_dampers {
            b.dampers.push(Damper {
                panels: bm.ranges(Part::FixedDamper(k)),
                epsilon: water[k].real("damping_factor")?,
                condition: DamperCondition::FreeImpervious,
                index: k as i32,
                name: water[k].name().to_string(),
                label: Part::FixedDamper(k).text(o),
            });
        }
        for g in o.children("generalized_modes") {
            let mut q = GeneralizedMode { name: g.name().to_string(), ..GeneralizedMode::default() };
            q.parity = index_of(&MODE_NAMES, g.choice("parity")?) + 1;
            for pc in g.children("pieces") {
                let mut p = ModePiece::default();
                // no ranges: every element (of the body's mesh)
                let r = MeshRef { name: own.clone(), ranges: ranges_item(pc) };
                if !r.ranges.is_empty() {
                    for x in &r.ranges {
                        if x.last > bm.stored {
                            return Err(Error::value(format!(
                                "{}ranges: the element range ({}, {}) is beyond the mesh {} ({} elements)",
                                pc.prefix(),
                                x.first,
                                x.last,
                                py_str_repr(&own),
                                bm.stored
                            )));
                        }
                    }
                    p.panels = ranges_of(&bm.records(&r));
                    if p.panels.is_empty() {
                        return Err(Error::value(format!(
                            "{}ranges: the piece applies to no element of the body",
                            pc.prefix()
                        )));
                    }
                }
                const COMPS: [&str; 3] = ["u", "v", "w"];
                for c in 0..3 {
                    let t = pc.reals(COMPS[c])?;
                    let mut k = 0usize;
                    while k + 4 <= t.len() {
                        p.terms.push(ModeTerm {
                            component: c as i32,
                            px: t[k + 1] as i32,
                            py: t[k + 2] as i32,
                            pz: t[k + 3] as i32,
                            c: t[k],
                        });
                        k += 4;
                    }
                }
                q.pieces.push(p);
            }
            b.generalized.push(q);
        }
        if let Some(v) = o.optional_reals("mode_mass") {
            b.mode_mass = v.to_vec();
        }
        if let Some(v) = o.optional_reals("mode_damping") {
            b.mode_damping = v.to_vec();
        }
        if let Some(v) = o.optional_reals("mode_stiffness") {
            b.mode_stiffness = v.to_vec();
        }
        for (k, t) in tanks.iter().enumerate() {
            let fs = t.real_or_name("surface_level")?;
            // the walls, the sheets inside and the plates of its free surface on the body's mesh: the
            // elements of the tank problem
            let mut recs: Vec<i32> = Vec::new();
            for (p, r) in &bm.parts {
                if matches!(p, Part::Tank(j) | Part::TankDipole(j, _) | Part::TankDamper(j, _) | Part::TankPlate(j, _) if *j == k)
                {
                    recs.extend(bm.records(r));
                }
            }
            recs.sort();
            // the plates of its free surface (their meshes of their own after the body's others)
            let mut plates = Vec::new();
            for (s, fp) in t.children("free_surface_plates").iter().enumerate() {
                let mesh = match other(&MeshRef::of(*fp)) {
                    Some(r) => self.mesh_index(&r, &fp.prefix(), here)?,
                    None => -1,
                };
                plates.push(Plate {
                    panels: if mesh >= 0 { Vec::new() } else { bm.ranges(Part::TankPlate(k, s)) },
                    mesh: some_index(mesh),
                    name: fp.name().to_string(),
                    place: fp.position() as i32,
                });
            }
            b.tanks.push(Tank {
                panels: ranges_of(&recs),
                density: t.real("density_ratio")?,
                surface_level: fs, // walls_top: None
                name: t.name().to_string(),
                place: t.position() as i32,
                plates,
            });
        }
        for (s, ps) in o.children("pressure_surfaces").iter().enumerate() {
            let mut p = PressureSurface {
                panels: bm.ranges(Part::Pressure(s)),
                name: ps.name().to_string(),
                place: ps.position() as i32,
                ..PressureSurface::default()
            };
            p.mesh = some_index(pressure_meshes[s]);
            for m in ps.children("pressure_modes") {
                p.modes.push(PressureMode {
                    name: m.name().to_string(),
                    shape: choice(&SHAPES, m.choice("shape")?),
                    stiffness: m.real("additional_stiffness")?,
                    damping: m.real("additional_damping")?,
                });
            }
            p.chamber_coupling = ps.boolean("chamber_coupling")?;
            p.convention = if ps.choice("response_convention")? == "variational" {
                RestoringConvention::Variational
            } else {
                RestoringConvention::NumericCompatibility
            };
            b.pressure.push(p);
        }
        for (s, fp) in o.children("free_surface_plates").iter().enumerate() {
            b.plates.push(Plate {
                panels: if plate_meshes[s] >= 0 { Vec::new() } else { bm.ranges(Part::Plate(s)) },
                mesh: some_index(plate_meshes[s]),
                name: fp.name().to_string(),
                place: fp.position() as i32,
            });
        }
        for (part, _) in &bm.parts {
            if let Part::Lid(li) = part {
                self.lids[*li] = (bi as i32, bm.ranges(*part));
            }
        }
        self.m.bodies.push(b);
        Ok(())
    }
}

/// check_objects, then the problem of the objects into m (emptied first): the environment, the
/// options and the first- and second-order switches, the bodies (each mesh added at its first use:
/// the body mesh, the interior, sheet, control-surface and pressure-surface meshes), the lids, the
/// fixed dampers, the frequencies and the field points (of the sea, then of the tanks).
pub fn build_model(doc: &Document, m: &mut Problem) -> Result<()> {
    check_objects(doc)?;
    Builder { doc, m, meshes: Vec::new(), lids: Vec::new() }.run()?;
    Ok(())
}

/// The problem of the copy of a model for a scope (`document_enabled::scoped_document`) into m:
/// - `Scope::Model`: `build_model`;
/// - `Scope::Body`: the rules of its bodies, lids and fixed dampers (`check_body_objects`), then
///   the build;
/// - `Scope::FreeSurface`: the rules of the items of `second_order` that choose others (its
///   free-surface items: a method needs its items) and those of the bodies, lids and fixed dampers
///   (`check_body_objects`), then the build, without the field points (no input of the free
///   surface); the rules of the frequencies, the orders' switches and the other options are not
///   applied;
/// - `Scope::BodyMesh`: no problem (the mesh is built from the copy by `build_body_mesh`); an
///   internal error.
pub fn build_scoped(doc: &Document, scope: Scope, m: &mut Problem) -> Result<()> {
    match scope {
        Scope::Model => return build_model(doc, m),
        Scope::Body(_) => check_body_objects(doc, None)?,
        Scope::FreeSurface => {
            if doc.root().children("bodies").is_empty() {
                return Err(Error::value("the model has no bodies"));
            }
            check_switched(single(doc.root(), "second_order"))?;
            check_body_objects(doc, None)?;
        }
        Scope::BodyMesh(_) => return Err(Error::internal("the scope of a body's mesh builds no problem")),
    }
    Builder { doc, m, meshes: Vec::new(), lids: Vec::new() }.run()?;
    if scope == Scope::FreeSurface {
        m.has_field_points = false;
        m.field_points.clear();
        m.field_tanks.clear();
    }
    Ok(())
}
