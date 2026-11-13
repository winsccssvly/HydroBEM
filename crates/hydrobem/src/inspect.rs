//! Model inspection: the panels and the tessellation of the surfaces (`Model::panels`,
//! `Model::tessellation`), the patch points of a mesh (`Model::mesh_patch_points`), and the
//! geometry and elements of a mesh alone (`Model::mesh_geometry`, `Model::mesh_elements`).
//!
//! One walk over the elements of a scope (every body, then the lids, fixed dampers and the free
//! surface; or one body) yields per element its key (body, role, tank, owner, element, part,
//! reflection) and its geometry; the panels take one row per element and the tessellation divides
//! the same elements, so that every quad has the key of one panels row.

use crate::codes::PanelRole;
use crate::error::{Error, Result};
use crate::geometry::{Panel, Placement, Reflection, Vec3, cross, norm, reflect_vertices, reflections, v3};
use crate::hydrostatics::Acc;
use crate::interior::interior_panels;
use crate::mesh::{Mesh, record_eval};
use crate::mesh_desc::PatchKind;
use crate::model_codes::InteriorMethod;
use crate::patch_geometry::inside_point;
use crate::problem::{Body, BodyDerived, Problem, Role};
use crate::quadrature::record_quadrature;

// the point and unit normal into the fluid of a patch at (u, v), mesh coordinates: a flat
// patch is the bilinear map of its corners with x_u x x_v into the fluid, the spline,
// analytic and trimmed maps have x_u x x_v into the body (quadrature.rs)
fn patch_point(m: &Mesh, rec: i32, u: f64, v: f64) -> (Vec3, Vec3) {
    let (x, n) = patch_normal(m, rec, u, v);
    if n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0 {
        return (x, n);
    }
    // a degenerate point of the map (an axis, a collapsed side): the normal just inside the patch
    let (ui, vi) = inside_point(u, v);
    let (_, n) = patch_normal(m, rec, ui, vi);
    (x, n)
}

// the point and the unit normal of patch_point at (u, v) itself (zero at a degenerate point)
fn patch_normal(m: &Mesh, rec: i32, u: f64, v: f64) -> (Vec3, Vec3) {
    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    record_eval(m, rec, u, v, &mut x, &mut xu, &mut xv);
    let flat = m.geometry == PatchKind::Flat && !m.trimmed();
    let mut n = if flat { cross(xu, xv) } else { -1.0 * cross(xu, xv) };
    let a = norm(n);
    n = if a > 0.0 { (1.0 / a) * n } else { v3(0.0, 0.0, 0.0) };
    (x, n)
}

/// What a query of the panels or of the tessellation covers: every body and then the elements of
/// no body (the lids, fixed dampers and the free surface), or one body (zero-based).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    All,
    Body(i32),
}

impl Scope {
    // the bodies of the scope, in order
    fn bodies(self, m: &Problem) -> Result<Vec<i32>> {
        let n = m.bodies.len() as i32;
        Ok(match self {
            Scope::All => (0..n).collect(),
            Scope::Body(b) if (0..n).contains(&b) => vec![b],
            Scope::Body(b) => return Err(Error::argument(format!("body index {b} does not exist"))),
        })
    }
    fn water(self) -> bool {
        matches!(self, Scope::All)
    }
}

fn reflected(x: &Vec3, r: &Reflection) -> Vec3 {
    v3(r.sx * x[0], r.sy * x[1], x[2])
}

/// The elements of the bodies, the lids, the fixed dampers and the free surface as the engine uses
/// them (`Model::panels`, `Model::body_panels`), for display: one row per element or reflected
/// copy. The seven keys (body, role, tank, owner, element, part, reflection) identify one row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Panels {
    /// the corners, global (a triangle repeats a vertex; a patch its corners), m
    pub vertices: Vec<[[f64; 3]; 4]>,
    /// the distinct vertices among the four corners: 3 (a triangle) or 4
    pub distinct_vertices: Vec<i32>,
    /// the zero-based body (among the enabled bodies); -1 an element of no body (a lid not placed
    /// with a body, a fixed damper, the free surface)
    pub body: Vec<i32>,
    /// the role of the element
    pub role: Vec<PanelRole>,
    /// the zero-based tank of its body whose fluid the element bounds or lies in (the walls, the
    /// dipoles and dipole dampers of a tank), else -1
    pub tank: Vec<i32>,
    /// the zero-based input object that makes the element, by its place in its list: a dipole in
    /// the dipoles of the body (or of its tank, `tank` set), a dipole damper likewise, a pressure
    /// surface, a free-surface plate, a lid in the model's lids, a fixed damper in the model's
    /// fixed dampers, the walls of a tank its tank; -1 for wetted, interior_surface,
    /// control_surface and free_surface
    pub owner: Vec<i32>,
    /// the one-based element in the numbering of the `ranges` of the mesh its input names (the
    /// stored elements); of an element a trim cut, the element it was cut from; of a surface the
    /// engine generates, its own numbering from 1 in the order of generation
    pub element: Vec<i32>,
    /// the zero-based part of an element a trim cut (in the order of the trim), 0 for an element
    /// not cut
    pub part: Vec<i32>,
    /// the copy of the stored element about the symmetry planes of its mesh (sx, sy), each 1 or -1;
    /// of a trimmed body the copy the cut element came from
    pub reflection: Vec<[i32; 2]>,
    /// the centroid (a patch: its point at the centre of its parameters), global, m
    pub centroids: Vec<[f64; 3]>,
    /// the unit normal into the fluid at the centroid (tank walls into the tank fluid), global
    pub normals: Vec<[f64; 3]>,
    /// the area, m^2
    pub areas: Vec<f64>,
}

impl Panels {
    /// The number of rows.
    pub fn len(&self) -> usize {
        self.areas.len()
    }
    pub fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }
}

/// The stored elements of a mesh alone, in its own coordinates, without their reflections
/// (`Model::mesh_elements`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshElements {
    /// the one-based element (the numbering of the mesh references)
    pub element: Vec<i32>,
    /// the corners (a triangle repeats a vertex; a patch its corners), m
    pub vertices: Vec<[[f64; 3]; 4]>,
    /// the distinct vertices among the four corners: 3 or 4
    pub distinct_vertices: Vec<i32>,
    /// the centroid (a patch: its point at the centre of its parameters), m
    pub centroids: Vec<[f64; 3]>,
    /// the unit normal into the fluid at the centroid
    pub normals: Vec<[f64; 3]>,
    /// the area, m^2
    pub areas: Vec<f64>,
}

impl MeshElements {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.areas.len()
    }
    pub fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }
}

/// A quadrilateral mesh of surfaces for display (`Model::tessellation`, `Model::body_tessellation`,
/// `Model::mesh_tessellation`): one row per quadrilateral, with the keys of the panels row of the
/// element it lies on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tessellation {
    /// the corners, global (of a mesh alone: mesh coordinates), m
    pub vertices: Vec<[[f64; 3]; 4]>,
    /// the unit normals into the fluid at the corners
    pub normals: Vec<[[f64; 3]; 4]>,
    /// the body, as `Panels::body` (-1 also for a mesh alone)
    pub body: Vec<i32>,
    /// the role, as `Panels::role`; None for a mesh alone
    pub role: Vec<Option<PanelRole>>,
    /// the tank, as `Panels::tank`
    pub tank: Vec<i32>,
    /// the owner, as `Panels::owner`
    pub owner: Vec<i32>,
    /// the one-based element, as `Panels::element` (of a mesh alone its stored element)
    pub element: Vec<i32>,
    /// the part, as `Panels::part`
    pub part: Vec<i32>,
    /// the reflection (sx, sy), as `Panels::reflection`
    pub reflection: Vec<[i32; 2]>,
}

impl Tessellation {
    /// The number of quadrilaterals.
    pub fn len(&self) -> usize {
        self.body.len()
    }
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }
}

// ---------------------------------------------------------------- the walk

/// The key of an element row (`Panels`, `Tessellation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    body: i32,
    role: Option<PanelRole>,
    tank: i32,
    owner: i32,
    element: i32,
    part: i32,
    reflection: [i32; 2],
}

/// The geometry of an element: a zero-based element of a mesh, a low-order panel of a generated
/// surface, or a panel of the free surface (global, in z = 0).
#[derive(Clone, Copy)]
enum Geom<'a> {
    Rec(&'a Mesh, i32),
    Panel(&'a Panel),
    Free(&'a [[f64; 3]; 4]),
}

/// One element of the walk: its key, its geometry in its mesh's coordinates, the placement of
/// those coordinates and the reflection of the stored element.
struct Element<'a> {
    key: Key,
    geom: Geom<'a>,
    place: Placement,
    refl: Reflection,
}

// the role, tank and owner of a zero-based element of the body's engine mesh
fn role_of(b: &Body, d: &BodyDerived, rec: i32) -> (PanelRole, i32, i32) {
    let r = rec as usize;
    let tank = if d.tank.is_empty() { -1 } else { d.tank[r] };
    let (role, owner) = match d.role[r] {
        Role::Wet => (PanelRole::Wetted, -1),
        Role::Interior => (PanelRole::InteriorSurface, -1),
        Role::Pressure => (PanelRole::PressureSurface, d.owner[r]),
        Role::Lid => (PanelRole::Lid, d.owner[r]),
        Role::Plate => (PanelRole::FreeSurfacePlate, d.owner[r]),
        // a wall of a tank, or a plate of its free surface (its place in the tank's plates)
        Role::Tank if d.owner[r] >= 0 => (PanelRole::FreeSurfacePlate, d.owner[r]),
        Role::Tank => (PanelRole::Tank, tank),
        Role::Dipole => {
            // a dipole damper of the body or its tank, a fixed damper placed in the body's frame,
            // or a dipole of the body or its tank
            let holds = |p: &[crate::mesh::Range]| p.iter().any(|rg| rec + 1 >= rg.first && rec < rg.last);
            match b.dampers.iter().find(|x| holds(&x.panels)) {
                Some(x) => (if x.free() { PanelRole::FixedDamper } else { PanelRole::DipoleDamper }, x.index),
                None => (PanelRole::Dipole, b.dipole_owners.iter().find(|x| holds(&x.panels)).map_or(-1, |x| x.index)),
            }
        }
    };
    (role, tank, owner)
}

/// The mesh of lid l in global coordinates: its own, or the one the higher order appended to the
/// first body's mesh (kept as given).
fn global_lid_mesh(m: &Problem, l: usize) -> Option<i32> {
    let lid = &m.lids[l];
    if let Some(mi) = lid.mesh {
        return Some(mi as i32);
    }
    let b = m.bodies.first()?;
    m.joined(b.mesh).iter().find(|j| j.lid == l as i32).map(|j| j.mesh)
}

// the reflections of a mesh's symmetry flags (only the stored copy without include_reflections)
fn copies(isx: i32, isy: i32, include_reflections: bool) -> Vec<Reflection> {
    let mut refl = reflections(isx, isy);
    if !include_reflections {
        refl.truncate(1);
    }
    refl
}

/// The default roles of the queries: every role but the two auxiliary surfaces (the control
/// surface and the free surface, no boundaries of the problem).
fn wanted(keep: &[PanelRole], role: PanelRole) -> bool {
    if keep.is_empty() {
        !matches!(role, PanelRole::ControlSurface | PanelRole::FreeSurface)
    } else {
        keep.contains(&role)
    }
}

// the body, role, tank and owner of a zero-based element of a mesh (None: not in the rows)
type KeyOf<'a> = &'a dyn Fn(i32) -> Option<(i32, PanelRole, i32, i32)>;

// what a walk keeps: the roles (empty: the default ones) and whether the reflections
#[derive(Clone, Copy)]
struct Keep<'a> {
    roles: &'a [PanelRole],
    include_reflections: bool,
}

// the elements of mesh mi of the problem placed by place, by their keys
fn mesh_walk(m: &Problem, mi: i32, place: Placement, keep: Keep, key: KeyOf, f: &mut dyn FnMut(&Element)) {
    let mesh = &m.meshes[mi as usize];
    for r in copies(mesh.isx, mesh.isy, keep.include_reflections) {
        for rec in 0..mesh.count {
            let Some((body, role, tank, owner)) = key(rec) else { continue };
            if !wanted(keep.roles, role) {
                continue;
            }
            let (element, part, copy) = m.element_key(mi, rec);
            let reflection = [copy[0] * r.sx as i32, copy[1] * r.sy as i32];
            let key = Key { body, role: Some(role), tank, owner, element, part, reflection };
            f(&Element { key, geom: Geom::Rec(mesh, rec), place, refl: r });
        }
    }
}

/// The walk over the elements of the scope, in order: per body the elements of its mesh (the lids
/// placed with it among them; not the fixed dampers on them, which are elements of no body), then
/// its surfaces of other meshes (dipoles, interior surface, pressure surfaces, plates, the meshes
/// the higher order joined to its mesh listed as those meshes), the control surface, the generated
/// interior surface (low order); then the lids not placed with a body, the fixed dampers (on a
/// body's mesh, then of their own) and the free surface of the potential QTF. keep: the roles kept
/// (empty: the default roles); with include_reflections every copy about the symmetry planes,
/// else the stored copy.
fn walk(
    m: &Problem,
    scope: Scope,
    keep: &[PanelRole],
    include_reflections: bool,
    f: &mut dyn FnMut(&Element),
) -> Result<()> {
    let k = Keep { roles: keep, include_reflections };
    for bi in scope.bodies(m)? {
        let b = &m.bodies[bi as usize];
        let mesh = &m.meshes[b.mesh as usize];
        let d = &m.derived()[bi as usize];
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        // the elements of the body's own mesh (those of other meshes that the higher order joined
        // follow as those meshes; the fixed dampers on them are elements of no body)
        mesh_walk(
            m,
            b.mesh,
            place,
            k,
            &|rec| {
                if m.joined_at(b.mesh, rec).is_some() {
                    return None;
                }
                let (role, tank, owner) = role_of(b, d, rec);
                (role != PanelRole::FixedDamper).then_some((bi, role, tank, owner))
            },
            f,
        );
        // a joined mesh by the role of its first element in the body's mesh (one role per mesh)
        let joined_as = |role: Role| {
            m.joined(b.mesh)
                .iter()
                .filter(move |j| j.lid < 0 && d.role[j.first as usize] == role)
                .map(|j| (j.mesh, d.owner[j.first as usize]))
                .collect::<Vec<_>>()
        };
        let dipole_owner = |rec: i32| b.dipole_mesh_owner.get(rec as usize).copied().unwrap_or(-1);
        let mut dipole_meshes: Vec<i32> = joined_as(Role::Dipole).into_iter().map(|j| j.0).collect();
        if let Some(dm) = b.dipole_mesh {
            dipole_meshes.insert(0, dm as i32);
        }
        for mi in dipole_meshes {
            mesh_walk(m, mi, place, k, &|rec| Some((bi, PanelRole::Dipole, -1, dipole_owner(rec))), f);
        }
        let mut interior: Vec<i32> = joined_as(Role::Interior).into_iter().map(|j| j.0).collect();
        if b.interior_method == InteriorMethod::ExternalMesh {
            interior
                .insert(0, b.interior_mesh.expect("an external interior surface has its mesh (rule_bodies)") as i32);
        }
        for mi in interior {
            mesh_walk(m, mi, place, k, &|_| Some((bi, PanelRole::InteriorSurface, -1, -1)), f);
        }
        let mut pressure: Vec<(i32, i32)> =
            b.pressure.iter().enumerate().filter_map(|(s, p)| p.mesh.map(|pi| (pi as i32, s as i32))).collect();
        pressure.extend(joined_as(Role::Pressure));
        for (mi, s) in pressure {
            mesh_walk(m, mi, place, k, &|_| Some((bi, PanelRole::PressureSurface, -1, s)), f);
        }
        let mut plates: Vec<(i32, i32)> =
            b.plates.iter().enumerate().filter_map(|(s, p)| p.mesh.map(|pi| (pi as i32, s as i32))).collect();
        plates.extend(joined_as(Role::Plate));
        for (mi, s) in plates {
            mesh_walk(m, mi, place, k, &|_| Some((bi, PanelRole::FreeSurfacePlate, -1, s)), f);
        }
        // the plates of the tanks' free surfaces on meshes of their own (body coordinates)
        for (ti, t) in b.tanks.iter().enumerate() {
            for (s, pi) in t.plates.iter().enumerate().filter_map(|(s, p)| p.mesh.map(|pi| (s, pi))) {
                let key = Some((bi, PanelRole::FreeSurfacePlate, ti as i32, s as i32));
                mesh_walk(m, pi as i32, place, k, &|_| key, f);
            }
        }
        if let Some(cs) = b.control_surface
            && wanted(keep, PanelRole::ControlSurface)
        {
            mesh_walk(m, cs as i32, place, k, &|_| Some((bi, PanelRole::ControlSurface, -1, -1)), f);
        }
        let generated =
            matches!(b.interior_method, InteriorMethod::Projected | InteriorMethod::Waterline | InteriorMethod::Graded);
        if generated && wanted(keep, PanelRole::InteriorSurface) && !mesh.patch {
            let ip = interior_panels(m, bi)?;
            for r in copies(ip.isx, ip.isy, include_reflections) {
                for (k, p) in ip.panels.iter().enumerate() {
                    let key = Key {
                        body: bi,
                        role: Some(PanelRole::InteriorSurface),
                        tank: -1,
                        owner: -1,
                        element: k as i32 + 1,
                        part: 0,
                        reflection: [r.sx as i32, r.sy as i32],
                    };
                    f(&Element { key, geom: Geom::Panel(p), place, refl: r });
                }
            }
        }
    }
    if !scope.water() {
        return Ok(());
    }
    let origin = Placement::default();
    for l in 0..m.lids.len() {
        if let Some(mi) = global_lid_mesh(m, l) {
            mesh_walk(m, mi, origin, k, &|_| Some((-1, PanelRole::Lid, -1, l as i32)), f);
        }
    }
    // the fixed dampers placed on the elements of a body's mesh (in the body's frame)
    if wanted(keep, PanelRole::FixedDamper) {
        for (bi, b) in m.bodies.iter().enumerate() {
            let d = &m.derived()[bi];
            let place = Placement { position: b.position, yaw_deg: b.yaw };
            let key = |rec| {
                let (role, tank, owner) = role_of(b, d, rec);
                (role == PanelRole::FixedDamper && m.joined_at(b.mesh, rec).is_none())
                    .then_some((-1, role, tank, owner))
            };
            mesh_walk(m, b.mesh, place, k, &key, f);
        }
    }
    for x in &m.fixed_dampers {
        mesh_walk(m, x.mesh, origin, k, &|_| Some((-1, PanelRole::FixedDamper, -1, x.index)), f);
    }
    if !keep.contains(&PanelRole::FreeSurface) {
        return Ok(());
    }
    // the panelled zone of the potential QTF's free surface as the solve integrates it
    // (second_order.free_surface_*; theory §12.8), body -1
    let z = crate::potential_qtf::zones(m)?;
    if z.panels.is_empty() {
        return Err(Error::value(
            "second_order: the generated cells have no panels (their points are the query of the free surface's points)",
        ));
    }
    // a given mesh keyed as every input mesh (its stored elements, the copies about its symmetry
    // planes, reflected quads reversed); the generated panels numbered from 1 in their order
    if let Some(fm) = m.opt.free_surface_mesh {
        mesh_walk(m, fm as i32, origin, k, &|_| Some((-1, PanelRole::FreeSurface, -1, -1)), f);
        return Ok(());
    }
    let one = Reflection { sx: 1.0, sy: 1.0 };
    for (k, v) in z.panels.iter().enumerate() {
        let key = Key {
            body: -1,
            role: Some(PanelRole::FreeSurface),
            tank: -1,
            owner: -1,
            element: k as i32 + 1,
            part: 0,
            reflection: [1, 1],
        };
        f(&Element { key, geom: Geom::Free(v), place: origin, refl: one });
    }
    Ok(())
}

// ---------------------------------------------------------------- the panels

// the rows of the panels, flat
#[derive(Default)]
struct PanelRows {
    verts: Vec<f64>,
    cents: Vec<f64>,
    nrms: Vec<f64>,
    areas: Vec<f64>,
    counts: Vec<i32>,
    keys: Vec<Key>,
}

impl PanelRows {
    // the row of an element: its corners, centroid, normal (its mesh coordinates, reflected) and
    // area, placed
    fn add(&mut self, e: &Element) {
        let r = &e.refl;
        let (v, c, n, area, nv) = match e.geom {
            Geom::Rec(mesh, rec) if !mesh.patch => {
                let p = &mesh.panels[rec as usize];
                (reflect_vertices(&p.proj, r), reflected(&p.centroid, r), reflected(&p.normal, r), p.area, p.distinct)
            }
            Geom::Rec(mesh, rec) => {
                let uv = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
                let mut v = [Vec3::ZERO; 4];
                for k in 0..4 {
                    v[k] = reflected(&patch_point(mesh, rec, uv[k][0], uv[k][1]).0, r);
                }
                if r.sx * r.sy < 0.0 {
                    v.reverse(); // keep the orientation
                }
                let (c, n) = patch_point(mesh, rec, 0.0, 0.0);
                let mut area = 0.0;
                for q in record_quadrature(mesh, rec, &Reflection { sx: 1.0, sy: 1.0 }) {
                    area += norm(q.nds);
                }
                (v, reflected(&c, r), reflected(&n, r), area, 4)
            }
            Geom::Panel(p) => {
                (reflect_vertices(&p.proj, r), reflected(&p.centroid, r), reflected(&p.normal, r), p.area, p.distinct)
            }
            Geom::Free(q) => {
                // the centroid and area of the panel as the solve integrates it
                let (c, a) = crate::free_surface_panels::centroid_area(&q.map(|x| [x[0], x[1]]));
                let tri = q[2] == q[3];
                (q.map(Vec3), v3(c[0], c[1], 0.0), v3(0.0, 0.0, 1.0), a.abs(), if tri { 3 } else { 4 })
            }
        };
        for x in &v {
            self.verts.extend_from_slice(&e.place.apply(*x).0);
        }
        self.cents.extend_from_slice(&e.place.apply(c).0);
        self.nrms.extend_from_slice(&e.place.rotate(n).0);
        self.areas.push(area);
        self.counts.push(nv);
        self.keys.push(e.key);
    }

    fn panels(self) -> Panels {
        let k = &self.keys;
        Panels {
            vertices: corners(&self.verts),
            distinct_vertices: self.counts,
            body: k.iter().map(|k| k.body).collect(),
            role: k.iter().map(|k| k.role.expect("a role of a model's element")).collect(),
            tank: k.iter().map(|k| k.tank).collect(),
            owner: k.iter().map(|k| k.owner).collect(),
            element: k.iter().map(|k| k.element).collect(),
            part: k.iter().map(|k| k.part).collect(),
            reflection: k.iter().map(|k| k.reflection).collect(),
            centroids: triples(&self.cents),
            normals: triples(&self.nrms),
            areas: self.areas,
        }
    }
}

// the rows of flat [N][4][3] corners
fn corners(v: &[f64]) -> Vec<[[f64; 3]; 4]> {
    v.chunks_exact(12).map(|c| std::array::from_fn(|k| [c[3 * k], c[3 * k + 1], c[3 * k + 2]])).collect()
}

// the rows of flat [N][3] values
fn triples(v: &[f64]) -> Vec<[f64; 3]> {
    v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// The elements of the scope as the engine uses them, after the model rules (the order of `walk`):
/// one row per element and copy kept; roles: the roles kept (empty: every role but the control
/// surface and the free surface).
pub fn inspect_panels(m: &Problem, scope: Scope, roles: &[PanelRole], include_reflections: bool) -> Result<Panels> {
    let mut rows = PanelRows::default();
    walk(m, scope, roles, include_reflections, &mut |e| rows.add(e))?;
    Ok(rows.panels())
}

/// The stored elements of a mesh alone, mesh coordinates, without their reflections: per element
/// its vertices (a patch: its corners), distinct vertex count, centroid (a patch: the point at its
/// centre), unit normal into the fluid and area.
pub fn inspect_elements(mesh: &Mesh) -> MeshElements {
    let mut rows = PanelRows::default();
    let one = Reflection { sx: 1.0, sy: 1.0 };
    for rec in 0..mesh.count {
        let key = Key { body: -1, role: None, tank: -1, owner: -1, element: rec + 1, part: 0, reflection: [1, 1] };
        rows.add(&Element { key, geom: Geom::Rec(mesh, rec), place: Placement::default(), refl: one });
    }
    MeshElements {
        element: rows.keys.iter().map(|k| k.element).collect(),
        vertices: corners(&rows.verts),
        distinct_vertices: rows.counts,
        centroids: triples(&rows.cents),
        normals: triples(&rows.nrms),
        areas: rows.areas,
    }
}

// ---------------------------------------------------------------- the tessellation

// the quads of a tessellation: vertices, normals at the vertices, keys
#[derive(Default)]
struct Quads {
    verts: Vec<f64>,
    nrms: Vec<f64>,
    keys: Vec<Key>,
}

impl Quads {
    // the quads of an element: a low-order panel one, a patch du x dv on its exact surface
    // (divisions [0, 0]: one per span of its potential discretization)
    fn add(&mut self, e: &Element, divisions: [i32; 2]) {
        let r = &e.refl;
        let mut emit = |mut x: [Vec3; 4], mut n: [Vec3; 4]| {
            for k in 0..4 {
                x[k] = e.place.apply(reflected(&x[k], r));
                n[k] = e.place.rotate(reflected(&n[k], r));
            }
            if r.sx * r.sy < 0.0 {
                // a reflection of odd order: the same orientation reversed
                x.reverse();
                n.reverse();
            }
            for k in 0..4 {
                self.verts.extend_from_slice(&x[k].0);
                self.nrms.extend_from_slice(&n[k].0);
            }
            self.keys.push(e.key);
        };
        match e.geom {
            Geom::Rec(mesh, rec) if !mesh.patch => {
                let p = &mesh.panels[rec as usize];
                emit(p.proj, [p.normal; 4]);
            }
            Geom::Panel(p) => emit(p.proj, [p.normal; 4]),
            Geom::Free(q) => emit(q.map(Vec3), [v3(0.0, 0.0, 1.0); 4]),
            Geom::Rec(mesh, rec) => {
                let [du, dv] = if divisions == [0, 0] {
                    let q = mesh.discretization[rec as usize];
                    [q.nu.max(1), q.nv.max(1)]
                } else {
                    divisions
                };
                let (nu_, nv_) = (du as usize, dv as usize);
                let mut gx = vec![Vec3::ZERO; (nu_ + 1) * (nv_ + 1)];
                let mut gn = vec![Vec3::ZERO; gx.len()];
                for j in 0..=nv_ {
                    for i in 0..=nu_ {
                        (gx[j * (nu_ + 1) + i], gn[j * (nu_ + 1) + i]) = patch_point(
                            mesh,
                            rec,
                            -1.0 + 2.0 * i as f64 / du as f64,
                            -1.0 + 2.0 * j as f64 / dv as f64,
                        );
                    }
                }
                for j in 0..nv_ {
                    for i in 0..nu_ {
                        let a = j * (nu_ + 1) + i;
                        let c = a + 1;
                        let g = a + nu_ + 2;
                        let h = a + nu_ + 1;
                        emit([gx[a], gx[c], gx[g], gx[h]], [gn[a], gn[c], gn[g], gn[h]]);
                    }
                }
            }
        }
    }

    fn tessellation(self) -> Tessellation {
        let k = &self.keys;
        Tessellation {
            vertices: corners(&self.verts),
            normals: corners(&self.nrms),
            body: k.iter().map(|k| k.body).collect(),
            role: k.iter().map(|k| k.role).collect(),
            tank: k.iter().map(|k| k.tank).collect(),
            owner: k.iter().map(|k| k.owner).collect(),
            element: k.iter().map(|k| k.element).collect(),
            part: k.iter().map(|k| k.part).collect(),
            reflection: k.iter().map(|k| k.reflection).collect(),
        }
    }
}

/// The divisions of a tessellation: each in 1..256, or both 0 (the spans of each patch).
fn check_divisions(divisions: [i32; 2]) -> Result<()> {
    let ok = |n: i32| (1..=256).contains(&n);
    if divisions != [0, 0] && !(ok(divisions[0]) && ok(divisions[1])) {
        return Err(Error::value("divisions must be in 1..256 (or both 0: one quad per span of each patch)"));
    }
    Ok(())
}

/// A quadrilateral mesh of the elements of the scope for display, the elements of `inspect_panels`
/// with the same roles and include_reflections: a low-order panel one quad, a patch divisions[0] x
/// divisions[1] quads on its exact surface (both 0: one quad per span of its potential
/// discretization); each quad with the keys of its element.
pub fn inspect_tessellation(
    m: &Problem,
    scope: Scope,
    roles: &[PanelRole],
    divisions: [i32; 2],
    include_reflections: bool,
) -> Result<Tessellation> {
    check_divisions(divisions)?;
    let mut q = Quads::default();
    walk(m, scope, roles, include_reflections, &mut |e| q.add(e, divisions))?;
    Ok(q.tessellation())
}

/// The tessellation of a mesh alone, mesh coordinates: body, tank and owner -1, no role, the
/// stored element, part 0.
pub fn inspect_mesh_tessellation(mesh: &Mesh, divisions: [i32; 2], include_reflections: bool) -> Result<Tessellation> {
    check_divisions(divisions)?;
    let mut q = Quads::default();
    for r in copies(mesh.isx, mesh.isy, include_reflections) {
        for rec in 0..mesh.count {
            let key = Key {
                body: -1,
                role: None,
                tank: -1,
                owner: -1,
                element: rec + 1,
                part: 0,
                reflection: [r.sx as i32, r.sy as i32],
            };
            q.add(&Element { key, geom: Geom::Rec(mesh, rec), place: Placement::default(), refl: r }, divisions);
        }
    }
    Ok(q.tessellation())
}

/// The geometry of a mesh alone, in its own coordinates, over the whole mesh (its reflections
/// included): the volume closed by the mesh and the plane z = 0 (theory §3.2), the surface area, the
/// bounds and the waterplane at z = 0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshGeometry {
    /// the three estimates of the divergence theorem, the integrals of x n_x, y n_y and z n_z, m^3
    pub volumes: [f64; 3],
    /// their mean
    pub volume: f64,
    /// the centre of volume: the first moments of the volume divided by the z estimate (0 where that
    /// estimate is 0: an open sheet), m
    pub center_of_volume: [f64; 3],
    /// the surface area, m^2
    pub area: f64,
    /// the smallest and the largest coordinates of the vertices (a patch: of a 17 x 17 grid of its
    /// points), m
    pub bounds: [[f64; 3]; 2],
    /// the waterplane at z = 0: its area (the integral of -n_z), m^2
    pub waterplane_area: f64,
    /// its centroid (x, y) (0 without a waterplane: a submerged body), m
    pub waterplane_centroid: [f64; 2],
    /// its second moments about the origin: the integrals of x^2, y^2 and x y, m^4
    pub waterplane_moments: [f64; 3],
}

/// The geometry of a mesh alone (MeshGeometry): the integrals of the hydrostatics of a body at the
/// origin (the same quadrature) over every element and reflection.
pub fn mesh_geometry(mesh: &Mesh) -> MeshGeometry {
    let mut s = Acc::default();
    let mut area = 0.0;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut bound = |x: Vec3| {
        for c in 0..3 {
            lo[c] = lo[c].min(x[c]);
            hi[c] = hi[c].max(x[c]);
        }
    };
    for r in reflections(mesh.isx, mesh.isy) {
        for rec in 0..mesh.count {
            for q in record_quadrature(mesh, rec, &r) {
                s.add(q.x[0], q.x[1], q.x[2], q.nds, true);
                area += norm(q.nds);
            }
            if !mesh.patch {
                for x in &mesh.panels[rec as usize].raw {
                    bound(reflected(x, &r));
                }
            } else {
                for j in 0..=16 {
                    for i in 0..=16 {
                        let (u, v) = (-1.0 + i as f64 / 8.0, -1.0 + j as f64 / 8.0);
                        bound(reflected(&patch_normal(mesh, rec, u, v).0, &r));
                    }
                }
            }
        }
    }
    if mesh.count <= 0 {
        (lo, hi) = ([0.0; 3], [0.0; 3]);
    }
    // a moment over a zero volume or waterplane: 0 (not NaN)
    let per = |a: f64, b: f64| if b == 0.0 { 0.0 } else { a / b };
    MeshGeometry {
        volumes: [s.vx, s.vy, s.vz],
        volume: (s.vx + s.vy + s.vz) / 3.0,
        center_of_volume: [per(s.mx, s.vz), per(s.my, s.vz), per(s.mz, s.vz)],
        area,
        bounds: [lo, hi],
        waterplane_area: s.aw,
        waterplane_centroid: [per(s.ax, s.aw), per(s.ay, s.aw)],
        waterplane_moments: [s.axx, s.ayy, s.axy],
    }
}

/// Points and unit normals into the fluid of the one-based patch element at n parameter pairs uv
/// ([n][2]), mesh coordinates: the points and the normals, [n][3] each.
pub fn inspect_patch_points(mesh: &Mesh, element: i32, n: i32, uv: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    if !mesh.patch {
        return Err(Error::value("patch points need a higher-order (patch) mesh"));
    }
    if element < 1 || element > mesh.count {
        return Err(Error::value(format!("patch {element} does not exist ({} patches)", mesh.count)));
    }
    let k = 3 * n.max(0) as usize;
    let (mut points, mut normals) = (vec![0.0; k], vec![0.0; k]);
    for k in 0..n as usize {
        let u = uv[2 * k];
        let v = uv[2 * k + 1];
        if !((-1.0..=1.0).contains(&u) && (-1.0..=1.0).contains(&v)) {
            return Err(Error::value("patch parameters must be in [-1, 1]"));
        }
        let (x, nn) = patch_point(mesh, element - 1, u, v);
        for c in 0..3 {
            points[3 * k + c] = x[c];
            normals[3 * k + c] = nn[c];
        }
    }
    Ok((points, normals))
}
