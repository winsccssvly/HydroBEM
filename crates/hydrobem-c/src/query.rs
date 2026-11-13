//! The queries of a model (after its rules), of a body and of a mesh alone (reference/inputs-meshes §4, reference/queries):
//! each answer a table of the fields of the engine's answer. Points are given as [n, 3] reals (C
//! order), uv as [n, 2].

use std::ffi::c_char;

use hydrobem_core::{Hydrostatics, MeshElements, MeshGeometry, MeshQuality, Panels, Tessellation};

use crate::model::{HbModel, model};
use crate::object::{HB_NO_OBJECT, id};
use crate::status::{HbStatus, Out, elements, guard, opt_text, slice, text, writable};
use crate::table::{HbTable, give};

fn quads(v: &[[[f64; 3]; 4]]) -> Vec<f64> {
    v.as_flattened().as_flattened().to_vec()
}

/// The panels: vertices [n, 4, 3] (m, global), distinct_vertices, the keys body, role
/// (`hb_panel_roles`), tank, owner (0-based, -1 none), element (1-based), part (0-based),
/// reflection [n, 2], then centroids [n, 3] (m, global), normals [n, 3] (unit, into the fluid),
/// areas (m^2).
pub fn panels_table(p: &Panels) -> HbTable {
    let n = p.len();
    HbTable::new()
        .f64s("vertices", &[n, 4, 3], quads(&p.vertices))
        .i32s("distinct_vertices", &[n], p.distinct_vertices.clone())
        .i32s("body", &[n], p.body.clone())
        .i32s("role", &[n], p.role.iter().map(|r| r.value()).collect())
        .i32s("tank", &[n], p.tank.clone())
        .i32s("owner", &[n], p.owner.clone())
        .i32s("element", &[n], p.element.clone())
        .i32s("part", &[n], p.part.clone())
        .i32s("reflection", &[n, 2], p.reflection.as_flattened().to_vec())
        .f64s("centroids", &[n, 3], p.centroids.as_flattened().to_vec())
        .f64s("normals", &[n, 3], p.normals.as_flattened().to_vec())
        .f64s("areas", &[n], p.areas.clone())
}

fn tessellation_table(t: &Tessellation) -> HbTable {
    let n = t.len();
    HbTable::new()
        .f64s("vertices", &[n, 4, 3], quads(&t.vertices))
        .f64s("normals", &[n, 4, 3], quads(&t.normals))
        .i32s("body", &[n], t.body.clone())
        .i32s("role", &[n], t.role.iter().map(|r| r.map_or(-1, |r| r.value())).collect())
        .i32s("tank", &[n], t.tank.clone())
        .i32s("owner", &[n], t.owner.clone())
        .i32s("element", &[n], t.element.clone())
        .i32s("part", &[n], t.part.clone())
        .i32s("reflection", &[n, 2], t.reflection.as_flattened().to_vec())
}

// the roles of a role argument: null the default roles, else names separated by commas
fn roles_of(role: Option<&str>) -> Vec<&str> {
    role.map_or(Vec::new(), |r| r.split(',').map(str::trim).collect())
}

fn quality_table(q: &MeshQuality) -> HbTable {
    let (n, b) = (q.body.len(), q.volumes.len());
    HbTable::new()
        .i32s("body", &[n], q.body.clone())
        .i32s("element", &[n], q.element.clone())
        .i32s("part", &[n], q.part.clone())
        .f64s("size", &[n], q.size.clone())
        .f64s("aspect", &[n], q.aspect.clone())
        .f64s("per_wavelength", &[n], q.per_wavelength.clone())
        .i32s("flags", &[n], q.flags.iter().map(|f| f.0 as i32).collect())
        .f64s("volumes", &[b, 3], q.volumes.as_flattened().to_vec())
        .f64s("min_per_wavelength", &[b], q.min_per_wavelength.clone())
        .i32s("edges", &[b, 4], q.edges.as_flattened().to_vec())
}

fn elements_table(r: &MeshElements) -> HbTable {
    let n = r.len();
    HbTable::new()
        .i32s("element", &[n], r.element.clone())
        .f64s("vertices", &[n, 4, 3], quads(&r.vertices))
        .i32s("distinct_vertices", &[n], r.distinct_vertices.clone())
        .f64s("centroids", &[n, 3], r.centroids.as_flattened().to_vec())
        .f64s("normals", &[n, 3], r.normals.as_flattened().to_vec())
        .f64s("areas", &[n], r.areas.clone())
}

fn hydrostatics_table(h: &Hydrostatics) -> HbTable {
    HbTable::new()
        .f64s("volumes", &[3], h.volumes.to_vec())
        .f64("volume", h.volume)
        .f64("displaced_mass", h.displaced_mass)
        .f64s("center_of_buoyancy", &[3], h.cb.to_vec())
        .f64("waterplane_area", h.wp_area)
        .f64s("waterplane_centroid", &[2], h.wp_centroid.to_vec())
        .f64s("stiffness_rho_g", &[6, 6], h.c_rho_g.to_vec())
        .bool("has_weight", h.has_weight)
        .f64("mass", h.mass)
        .f64s("center_of_gravity", &[3], h.cg.to_vec())
        .f64s("stiffness", &[6, 6], h.c.to_vec())
        .f64s("metacentric_height", &[2], h.metacentric_height.to_vec())
}

fn geometry_table(g: &MeshGeometry) -> HbTable {
    HbTable::new()
        .f64s("volumes", &[3], g.volumes.to_vec())
        .f64("volume", g.volume)
        .f64s("center_of_volume", &[3], g.center_of_volume.to_vec())
        .f64("area", g.area)
        .f64s("bounds", &[2, 3], g.bounds.as_flattened().to_vec())
        .f64("waterplane_area", g.waterplane_area)
        .f64s("waterplane_centroid", &[2], g.waterplane_centroid.to_vec())
        .f64s("waterplane_moments", &[3], g.waterplane_moments.to_vec())
}

unsafe fn rows<const N: usize>(p: *const f64, n: usize, what: &str) -> Out<Vec<[f64; N]>> {
    let v = unsafe { slice(p, elements(&[n, N], what)?, what) }?;
    Ok(v.chunks_exact(N).map(|r| std::array::from_fn(|k| r[k])).collect())
}

/// The sizes of the problem: bodies, meshes, lids, ordinary_frequencies, headings, coordinates,
/// radiation_columns, field_points (in the sea), tank_points (in the tanks), lid_elements,
/// pressure_elements (with their reflections; `HB_I64`), has_zero_frequency,
/// has_infinite_frequency (0 or 1), radiation_mode_ids and per_lid_elements (`HB_I32` arrays), and
/// per enabled body (arrays of the bodies): stored, wetted, wetted_with_reflections, dipole,
/// interior_surface (-1 none), pressure_modes, symmetry_copies, generalized_modes.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_counts(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let (c, bodies, lids, modes) = unsafe { self::model(model) }?.with(|m| {
            let c = m.counts()?;
            Ok((c, m.solved_body_counts()?, m.lid_elements()?, m.radiation_mode_ids()?))
        })?;
        let nb = bodies.len();
        let per = |f: fn(&hydrobem_core::BodyCounts) -> i64| bodies.iter().map(f).collect::<Vec<_>>();
        let (nl, nm) = (lids.len(), modes.len());
        let t = HbTable::new()
            .i64("bodies", c.bodies as i64)
            .i64("meshes", c.meshes as i64)
            .i64("lids", c.lids as i64)
            .i64("ordinary_frequencies", c.ordinary_frequencies as i64)
            .i64("headings", c.headings as i64)
            .i64("coordinates", c.coordinates as i64)
            .i64("radiation_columns", c.radiation_columns as i64)
            .i64("field_points", c.field_points as i64)
            .i64("tank_points", c.tank_points as i64)
            .i64("lid_elements", c.lid_elements as i64)
            .i64("pressure_elements", c.pressure_elements as i64)
            .bool("has_zero_frequency", c.has_zero_frequency)
            .bool("has_infinite_frequency", c.has_infinite_frequency)
            .i32s("radiation_mode_ids", &[nm], modes)
            .i32s("per_lid_elements", &[nl], lids)
            .i64s("stored", &[nb], per(|b| b.stored as i64))
            .i64s("wetted", &[nb], per(|b| b.wetted as i64))
            .i64s("wetted_with_reflections", &[nb], per(|b| b.wetted_with_reflections as i64))
            .i64s("dipole", &[nb], per(|b| b.dipole as i64))
            .i64s("interior_surface", &[nb], per(|b| b.interior_surface.map_or(-1, |n| n as i64)))
            .i64s("pressure_modes", &[nb], per(|b| b.pressure_modes as i64))
            .i64s("symmetry_copies", &[nb], per(|b| b.symmetry_copies as i64))
            .i64s("generalized_modes", &[nb], per(|b| b.generalized_modes as i64));
        unsafe { give(out, t) }
    })
}

/// The motion equation (theory §3.1): inertia, stiffness and damping [d, d] and ready (0 or 1).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_motion_equation(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let e = unsafe { self::model(model) }?.with(|m| m.motion_equation())?;
        let n = [e.ndof, e.ndof];
        let t = HbTable::new()
            .f64s("inertia", &n, e.inertia)
            .f64s("stiffness", &n, e.stiffness)
            .f64s("damping", &n, e.damping)
            .bool("ready", e.ready);
        unsafe { give(out, t) }
    })
}

/// The elements as the engine uses them (after a trim) of every body and then of no body (the
/// lids not placed with a body, the fixed dampers, the free surface), of the roles of `role` (the
/// names of `hb_panel_roles` separated by commas, e.g. "wetted,lid"; null every role but
/// control_surface and free_surface); with their reflections when `include_reflections` is
/// nonzero. The seven keys (body, role, tank, owner, element, part, reflection) identify a row.
/// With `role` "free_surface" alone, the rules of the free surface only (as
/// `hb_model_free_surface_points` for the panelled zone); otherwise every rule of the model.
///
/// # Safety
/// `model` is a model of this library; `role` null or a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_panels(
    model: *mut HbModel,
    role: *const c_char,
    include_reflections: i32,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let roles = roles_of(unsafe { opt_text(role, "role") }?);
        let p = unsafe { self::model(model) }?.with(|m| m.panels(&roles, include_reflections != 0))?;
        unsafe { give(out, panels_table(&p)) }
    })
}

/// The points (global) and weights of one zone of the potential QTF's free surface as the solve
/// integrates it: `zone` "panelled" (the panels' Gauss points or the cells'), "quadrature" or
/// "outer_circle" (the line integral around the outer circle); points [n, 3] and weights [n]
/// (m^2, or m along the outer circle). It applies the rules of the free surface (its items, the
/// bodies' geometry and surfaces, the environment and its domain), not those of the frequencies,
/// the orders or the other options; a zone whose auto values the frequencies decide is refused
/// (`HB_ERR_VALUE`) without a frequency between 0 and inf.
///
/// # Safety
/// `model` is a model of this library; `zone` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_free_surface_points(
    model: *mut HbModel,
    zone: *const c_char,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let zone = unsafe { text(zone, "zone") }?;
        let p = unsafe { self::model(model) }?.with(|m| m.free_surface_points(zone))?;
        let n = p.points.len();
        let t =
            HbTable::new().f64s("points", &[n, 3], p.points.as_flattened().to_vec()).f64s("weights", &[n], p.weights);
        unsafe { give(out, t) }
    })
}

/// A quadrilateral mesh for display of the elements of `hb_model_panels` with the same `role`
/// and `include_reflections`: a low-order panel one quad, a patch divisions_u x divisions_v quads
/// on its exact surface (1..256 each; both 0: one quad per span of its potential discretization);
/// vertices and normals [n, 4, 3] (m, global; unit normals into the fluid at the corners) and the
/// keys of the panels row of each quad: body, role, tank, owner, element, part, reflection [n, 2].
/// The rules as `hb_model_panels` (with "free_surface" alone, those of the free surface).
///
/// # Safety
/// `model` is a model of this library; `role` null or a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_tessellation(
    model: *mut HbModel,
    role: *const c_char,
    divisions_u: i32,
    divisions_v: i32,
    include_reflections: i32,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let roles = roles_of(unsafe { opt_text(role, "role") }?);
        let d = [divisions_u, divisions_v];
        let t = unsafe { self::model(model) }?.with(|m| m.tessellation(&roles, d, include_reflections != 0))?;
        unsafe { give(out, tessellation_table(&t)) }
    })
}

/// The quality of the meshes of every body: per element body, element, part, size, aspect,
/// per_wavelength and flags (`HB_QUALITY_` bits); per body volumes [b, 3], min_per_wavelength and
/// edges [b, 4].
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_mesh_quality(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let q = unsafe { self::model(model) }?.with(|m| m.mesh_quality())?;
        unsafe { give(out, quality_table(&q)) }
    })
}

/// The displacements of every coordinate at n points of the bodies (the point in global
/// coordinates, its zero-based body and one-based stored element): values [n, d, 3].
///
/// # Safety
/// `model` is a model of this library; `points` 3 n reals; `bodies` and `elements` n integers;
/// `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_mode_displacements(
    model: *mut HbModel,
    points: *const f64,
    bodies: *const i32,
    elements: *const i32,
    n: usize,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let b = unsafe { slice(bodies, n, "bodies") }?;
        let r = unsafe { slice(elements, n, "elements") }?;
        let p = unsafe { rows::<3>(points, n, "points") }?;
        let (v, d) = unsafe { self::model(model) }?.with(|m| {
            let v = m.mode_displacements(&p, b, r)?;
            Ok((v, m.counts()?.coordinates))
        })?;
        unsafe { give(out, HbTable::new().f64s("values", &[n, d, 3], v)) }
    })
}

/// The memory of a solve in bytes: statics, per_frequency, outputs, peak (resident), commit.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_memory_estimate(model: *mut HbModel, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let e = unsafe { self::model(model) }?.with(|m| m.memory_estimate())?;
        let t = HbTable::new()
            .i64("statics", e.statics)
            .i64("per_frequency", e.per_frequency)
            .i64("outputs", e.outputs)
            .i64("peak", e.peak)
            .i64("commit", e.commit);
        unsafe { give(out, t) }
    })
}

/// The verdict of the field-point rules (reference/inputs-model-objects §3) on n points: `tank` 0 or
/// `HB_NO_OBJECT`, points in the sea (global coordinates); or the object of a tank of the model,
/// points inside it (body coordinates). Status (texts), body (the zero-based body of the status, or the body of the tank
/// of a point in a tank), lid (the lid a point is on) and tank (of a point in a tank the zero-based
/// tank of its body, as the results number a body's tanks); -1 none.
///
/// # Safety
/// `model` is a model of this library; `points` 3 n reals; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_classify_points(
    model: *mut HbModel,
    points: *const f64,
    n: usize,
    tank: u64,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let p = unsafe { rows::<3>(points, n, "points") }?;
        let tank = if tank == 0 || tank == HB_NO_OBJECT { None } else { Some(id(tank)) };
        let v = unsafe { self::model(model) }?.with(|m| m.classify_points(&p, tank))?;
        let status: Vec<&str> = v.iter().map(|c| c.status.name()).collect();
        let col = |f: fn(&hydrobem_core::PointClass) -> Option<usize>| {
            v.iter().map(|c| f(c).map_or(-1, |i| i as i64)).collect::<Vec<_>>()
        };
        let n = v.len();
        let t = HbTable::new()
            .texts("status", &status)
            .i64s("body", &[n], col(|c| c.body))
            .i64s("lid", &[n], col(|c| c.lid))
            .i64s("tank", &[n], col(|c| c.tank));
        unsafe { give(out, t) }
    })
}

/// The hydrostatics of a body after its trim and placement (theory §3.2): volumes [3] (the three
/// estimates of the divergence theorem), volume (their mean), displaced_mass, center_of_buoyancy [3]
/// (body coordinates), waterplane_area, waterplane_centroid [2] (body coordinates), stiffness_rho_g
/// [6, 6], has_weight, mass, center_of_gravity [3] (body coordinates), stiffness [6, 6] (about R),
/// metacentric_height [2].
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_body_hydrostatics(model: *mut HbModel, body: u64, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let h = unsafe { self::model(model) }?.with(|m| m.body_hydrostatics(id(body)))?;
        unsafe { give(out, hydrostatics_table(&h)) }
    })
}

/// The elements of a body as the engine uses them, as `hb_model_panels` (no fixed_damper or
/// free_surface, elements of no body).
///
/// # Safety
/// `model` is a model of this library; `role` null or a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_body_panels(
    model: *mut HbModel,
    body: u64,
    role: *const c_char,
    include_reflections: i32,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let roles = roles_of(unsafe { opt_text(role, "role") }?);
        let p = unsafe { self::model(model) }?.with(|m| m.body_panels(id(body), &roles, include_reflections != 0))?;
        unsafe { give(out, panels_table(&p)) }
    })
}

/// The quality of the mesh of a body, as `hb_model_mesh_quality`.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_body_mesh_quality(model: *mut HbModel, body: u64, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let q = unsafe { self::model(model) }?.with(|m| m.body_mesh_quality(id(body)))?;
        unsafe { give(out, quality_table(&q)) }
    })
}

/// The tessellation of the elements of a body as the engine uses them, placed, as
/// `hb_model_tessellation`.
///
/// # Safety
/// `model` is a model of this library; `role` null or a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_body_tessellation(
    model: *mut HbModel,
    body: u64,
    role: *const c_char,
    divisions_u: i32,
    divisions_v: i32,
    include_reflections: i32,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let roles = roles_of(unsafe { opt_text(role, "role") }?);
        let d = [divisions_u, divisions_v];
        let t = unsafe { self::model(model) }?
            .with(|m| m.body_tessellation(id(body), &roles, d, include_reflections != 0))?;
        unsafe { give(out, tessellation_table(&t)) }
    })
}

/// The geometry of a mesh alone, in its own coordinates: volumes [3], volume, center_of_volume
/// [3], area, bounds [2, 3], waterplane_area, waterplane_centroid [2], waterplane_moments [3].
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_geometry(model: *mut HbModel, mesh: u64, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let g = unsafe { self::model(model) }?.with(|m| m.mesh_geometry(id(mesh)))?;
        unsafe { give(out, geometry_table(&g)) }
    })
}

/// The stored elements of a mesh, mesh coordinates: element (1-based), vertices [n, 4, 3] (m),
/// distinct_vertices, centroids [n, 3] (m), normals [n, 3] (unit, into the fluid), areas (m^2).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_elements(model: *mut HbModel, mesh: u64, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::model(model) }?.with(|m| m.mesh_elements(id(mesh)))?;
        unsafe { give(out, elements_table(&r)) }
    })
}

/// The tessellation of a mesh alone, mesh coordinates (role -1, body, tank and owner -1, the
/// stored element, part 0), as `hb_model_tessellation`.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_tessellation(
    model: *mut HbModel,
    mesh: u64,
    divisions_u: i32,
    divisions_v: i32,
    include_reflections: i32,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let d = [divisions_u, divisions_v];
        let t = unsafe { self::model(model) }?.with(|m| m.mesh_tessellation(id(mesh), d, include_reflections != 0))?;
        unsafe { give(out, tessellation_table(&t)) }
    })
}

/// The geometric quality of a mesh alone (no wavelength), as `hb_model_mesh_quality`.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_geometry_quality(model: *mut HbModel, mesh: u64, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let q = unsafe { self::model(model) }?.with(|m| m.mesh_geometry_quality(id(mesh)))?;
        unsafe { give(out, quality_table(&q)) }
    })
}

/// Points and unit normals (into the fluid) of a patch (one-based element) at n parameters uv in
/// [-1, 1]² ([n, 2]), mesh coordinates: points [n, 3] (m), normals [n, 3].
///
/// # Safety
/// `model` is a model of this library; `uv` 2 n reals; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_patch_points(
    model: *mut HbModel,
    mesh: u64,
    element: i32,
    uv: *const f64,
    n: usize,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let uv = unsafe { rows::<2>(uv, n, "uv") }?;
        let p = unsafe { self::model(model) }?.with(|m| m.mesh_patch_points(id(mesh), element, &uv))?;
        let k = p.points.len();
        let t = HbTable::new().f64s("points", &[k, 3], p.points.as_flattened().to_vec()).f64s(
            "normals",
            &[k, 3],
            p.normals.as_flattened().to_vec(),
        );
        unsafe { give(out, t) }
    })
}
