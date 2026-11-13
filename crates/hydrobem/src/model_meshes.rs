//! The meshes of a model (model.rs) as objects: read from GDF, SPL, CSF and geometry files, made
//! from analytic shapes or the text of a geometry file, written as GDF (and SPL) or geometry
//! files; the queries of a mesh alone (its geometry, elements, tessellation, quality and patch
//! points); the control surfaces the engine makes. The trims are in model_trim.rs.

use std::path::Path;

use crate::analytic_shapes::{ShapeSpec, named_shape};
use crate::codes::ControlSurfaceKind;
use crate::constants::STANDARD_GRAVITY;
use crate::document::{ItemValue, Obj, ObjMut, ObjectId, check_value, schema_type, symmetry_flags};
use crate::document_build::{body_mesh as build_body_mesh, body_mesh_name, generated_plate, placed_fixed_dampers};
use crate::document_enabled::{Scope, scoped_document};
use crate::document_mesh::{fill_mesh, mesh_desc};
use crate::error::{Error, Result};
use crate::gdf::{
    GdfFile, PatchOptions, analytic_control_surface, analytic_desc, gdf_text, gdf_text_patch, read_csf,
    read_low_order_gdf, read_patch_gdf, spl_text,
};
use crate::hbg::{hbg_text, read_hbg};
use crate::inspect::{
    MeshElements, MeshGeometry, Tessellation, inspect_elements, inspect_mesh_tessellation, inspect_patch_points,
    mesh_geometry,
};
use crate::mesh::{Mesh, build_mesh, build_patch_mesh};
use crate::mesh_desc::{ControlSurfaceDesc, MeshKind, PatchDiscretization, PatchKind};
use crate::model::{Model, path_text};
use crate::model_types::{AnalyticShape, ControlSurfaceInput, MeshFile, PatchPoints};
use crate::preprocess::automatic_control_surface;
use crate::quality::{MeshQuality, geometry_quality};
use crate::text::{absolute_path, file_exists, make_parent_folder, read_file, write_file};

/// Whether a path names a geometry file (.hbg, any case).
fn is_hbg(path: &str) -> bool {
    Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("hbg"))
}

/// The reader options of a mesh file: its SPL file, a discretization or a panel size with the orders.
fn patch_options(file: &MeshFile, spl: Option<String>) -> PatchOptions {
    let mut o = PatchOptions::default();
    if let Some(s) = spl {
        o.spl = s;
    }
    if let Some(rows) = &file.discretization {
        o.has_discretization = true;
        for x in rows {
            o.discretization.push(PatchDiscretization {
                nu: x[0],
                nv: x[1],
                ku: x[2],
                kv: x[3],
                iquo: x[4],
                iqui: x[5],
            });
        }
    }
    o.has_panel_size = file.panel_size.is_some();
    o.panel_size = file.panel_size.unwrap_or(0.0);
    o.spline_order = file.spline_order.unwrap_or(0); // 0: the engine's rule
    o.outer_quadrature = file.outer_quadrature.unwrap_or(0);
    o.inner_quadrature = file.inner_quadrature.unwrap_or(0);
    o
}

/// The engine's mesh of a mesh object (validated).
pub(crate) fn mesh_of_object(o: Obj) -> Result<Mesh> {
    let d = mesh_desc(o)?;
    if d.low_order { build_mesh(&d.lo) } else { build_patch_mesh(&d.ho) }
}

pub(crate) fn need_mesh(o: Obj) -> Result<()> {
    if o.type_().name != "Mesh" {
        return Err(Error::argument(format!("{}is not a mesh", o.prefix())));
    }
    Ok(())
}

pub(crate) fn text_value(s: &str) -> ItemValue {
    ItemValue::Text(s.to_string())
}

pub(crate) fn real_value(x: f64) -> ItemValue {
    ItemValue::Real(x)
}

pub(crate) fn int_value(x: i64) -> ItemValue {
    ItemValue::Int(x)
}

pub(crate) fn real_array_value(shape: Vec<i64>, data: Vec<f64>) -> ItemValue {
    ItemValue::Reals { shape, data }
}

pub(crate) fn int_array_value(shape: Vec<i64>, data: Vec<i64>) -> ItemValue {
    ItemValue::Ints { shape, data }
}

// a body has a mesh: its wetted surface, or the mesh of its first dipole or free-surface plate
pub(crate) fn check_body_mesh(b: Obj) -> Result<()> {
    if body_mesh_name(b).is_none() && !b.children("free_surface_plates").first().is_some_and(|p| generated_plate(*p)) {
        return Err(Error::value(format!(
            "{}the body has no wetted surface, no dipoles and no free-surface plates",
            b.prefix()
        )));
    }
    Ok(())
}

// the engine mesh of a body object: its mesh as the problem uses it (trimmed, the elements its
// surfaces take; of the enabled objects, the body itself taken as enabled)
fn body_mesh(b: Obj) -> Result<Mesh> {
    if b.type_().name != "Body" {
        return Err(Error::argument(format!("{}is not a body", b.prefix())));
    }
    check_body_mesh(b)?;
    let e = scoped_document(b.document(), Scope::BodyMesh(b.id()))?;
    let b = e.doc.obj(b.id());
    let placed = placed_fixed_dampers(&e.doc)?;
    let bm = build_body_mesh(b, &placed[b.index()])?;
    let mut d = bm.desc;
    d.lo.gravity = STANDARD_GRAVITY;
    d.ho.gravity = STANDARD_GRAVITY;
    if d.low_order { build_mesh(&d.lo) } else { build_patch_mesh(&d.ho) }
}

/// The surface of a control surface the engine makes: a cylinder or an ellipsoid from the numbers
/// of a control-surface file, or the automatic surface around a body's mesh (with the free surface
/// in body axes).
pub(crate) fn control_surface_gdf(s: &ControlSurfaceInput, body: Option<(&Mesh, f64)>) -> Result<GdfFile> {
    control_surface_gdf_covered(s, body, Vec::new())
}

/// control_surface_gdf with the elements of the body's free-surface plates as (x, y) polygons in
/// body axes: the automatic surface leaves the waterline they cover out (ControlSurfaceDesc).
pub(crate) fn control_surface_gdf_covered(
    s: &ControlSurfaceInput,
    body: Option<(&Mesh, f64)>,
    covers: Vec<Vec<[f64; 2]>>,
) -> Result<GdfFile> {
    // the polygon and the partitions as (x, y) pairs
    let polygon: Vec<f64> = s.polygon.iter().flatten().copied().collect();
    let counts: Vec<i32> = s.partitions.iter().map(|p| p.len() as i32).collect();
    let partitions: Vec<f64> = s.partitions.iter().flatten().flatten().copied().collect();
    let mut g =
        GdfFile { higher_order: true, geometry: PatchKind::Analytic, label: s.label.clone(), ..GdfFile::default() };
    if s.kind == ControlSurfaceKind::Automatic {
        let Some((mesh, waterline_z)) = body else {
            return Err(Error::argument("the automatic control surface needs its body"));
        };
        // radius or polygon: checked by automatic_control_surface (preprocess.rs)
        let c = ControlSurfaceDesc {
            radius: s.radius,
            depth: s.depth,
            panel_size: s.panel_size,
            waterline_z,
            free_surface: s.free_surface as i32,
            polygon_count: (polygon.len() / 2) as i32,
            polygon,
            partitions,
            partition_counts: counts.to_vec(),
            partition_count: counts.len() as i32,
            covers,
        };
        let cs = automatic_control_surface(mesh, &c)?;
        for q in &cs.patches {
            g.analytic.push(analytic_desc(q));
        }
        g.discretization = cs.discretization;
        g.isx = cs.symmetry[0];
        g.isy = cs.symmetry[1];
    } else {
        analytic_control_surface(s.kind.icdef(), &s.values, s.panel_size, &mut g.analytic, &mut g.discretization)?;
        g.isx = 1;
        g.isy = 1;
    }
    g.count = g.analytic.len() as i32;
    Ok(g)
}

// the discretization of every patch: one row for all, or one per patch
fn discretization_rows(rows: &[[i32; 6]], npatch: i32) -> Result<Vec<PatchDiscretization>> {
    let n = rows.len() as i32;
    if n != 1 && n != npatch {
        return Err(Error::value(format!(
            "discretization has {n} rows for {npatch} patches (one for all, or one per patch)"
        )));
    }
    let mut out = Vec::new();
    for k in 0..npatch {
        let x = &rows[(if n == 1 { 0 } else { k }) as usize];
        out.push(PatchDiscretization { nu: x[0], nv: x[1], ku: x[2], kv: x[3], iquo: x[4], iqui: x[5] });
    }
    Ok(out)
}

impl Model {
    // a new mesh object filled by fill and validated; nothing is left behind when it fails
    pub(crate) fn new_mesh(&mut self, name: &str, fill: impl FnOnce(&mut ObjMut) -> Result<()>) -> Result<ObjectId> {
        let root = self.doc.root_id();
        let m = self.doc.add_child(root, "meshes", "Mesh", name, -1)?;
        let made = (|| {
            fill(&mut self.doc.obj_mut(m))?;
            mesh_of_object(self.doc.obj(m)).map(|_| ())
        })();
        let r = match made {
            Ok(()) => Ok(m),
            Err(e) => self.doc.remove_child(root, m).and(Err(e)),
        };
        self.sync();
        r
    }
    pub(crate) fn unique_mesh_name(&self, base: &str) -> String {
        let mut n = base.to_string();
        let mut k = 2;
        while self.doc.root().child("meshes", &n).is_some() {
            n = format!("{base}_{k}");
            k += 1;
        }
        n
    }
    /// Adds a mesh read from a GDF file (and its SPL file) by the rules of reference/inputs-meshes.
    pub fn load_mesh(&mut self, name: &str, file: &MeshFile) -> Result<ObjectId> {
        self.need_changeable()?;
        let path = path_text(&file.path)?.to_string();
        let spl = match &file.spl_path {
            Some(p) => Some(path_text(p)?.to_string()),
            None => None,
        };
        if is_hbg(&path) {
            // a geometry file: the mesh kind is the file's, its discretization the file's or, when
            // it has none, the one given here
            if spl.is_some() || file.zero_trailer {
                return Err(Error::argument("a geometry file (.hbg) is read without an SPL file or a zero trailer"));
            }
            let bytes = read_file(&path)?;
            let text = String::from_utf8(bytes).map_err(|_| Error::value(format!("{path}: the file is not UTF-8")))?;
            let o = patch_options(file, None);
            let at = format!("{path}: ");
            let src = absolute_path(&path);
            return self.new_mesh(name, |m| {
                read_hbg(m, &text, &at, &o, mesh_of_object)?;
                m.set("source_file", text_value(&src))
            });
        }
        let g: GdfFile = if file.higher_order {
            let o = patch_options(file, spl.clone());
            read_patch_gdf(&path, &o, file.zero_trailer)?
        } else {
            read_low_order_gdf(&path, file.zero_trailer)?
        };
        let higher = file.higher_order;
        self.new_mesh(name, |m| {
            fill_mesh(m, &g, true)?;
            // where it was read from, for the clients
            m.set("source_file", text_value(&absolute_path(&path)))?;
            if higher {
                if let Some(s) = &spl {
                    m.set("source_spl", text_value(&absolute_path(s)))?;
                }
            }
            Ok(())
        })
    }
    /// Adds the panels of a free-surface file (FDF) as a panels mesh in z = 0, with the symmetry
    /// of the model it belongs to ("none", "yz", "xz" or "xz_and_yz"), for second_order.free_surface_mesh.
    pub fn load_fdf(&mut self, name: &str, path: &Path, symmetry: &str) -> Result<ObjectId> {
        self.need_changeable()?;
        let p = path_text(path)?;
        let mut sym = text_value(symmetry);
        check_value(schema_type("Mesh").item("symmetry").expect("a Mesh has symmetry"), &mut sym)?;
        let flags = symmetry_flags(&sym);
        let g = crate::gdf::read_fdf(p, flags[0], flags[1])?;
        let source = absolute_path(p);
        self.new_mesh(name, |m| {
            fill_mesh(m, &g, false)?;
            m.set("source_file", text_value(&source))?; // where it was read from, for the clients
            Ok(())
        })
    }
    /// Adds a control surface read from a CSF file; body (or None) gives the automatic surface
    /// of the file its mesh and position.
    pub fn load_csf(
        &mut self,
        name: &str,
        path: &Path,
        zero_trailer: bool,
        body: Option<ObjectId>,
    ) -> Result<ObjectId> {
        self.need_changeable()?;
        let p = path_text(path)?;
        let mut bm: Option<Mesh> = None;
        let mut waterline = 0.0;
        if let Some(body) = body {
            let b = self.own_body(body)?;
            bm = Some(body_mesh(b)?);
            waterline = 0.0 - b.get("position").reals()[2]; // the free surface in body axes (+0.0 at z = 0)
        }
        let g = read_csf(p, zero_trailer, bm.as_ref(), waterline)?;
        let source = absolute_path(p);
        self.new_mesh(name, |m| {
            fill_mesh(m, &g, false)?;
            m.set("source_file", text_value(&source))?; // where it was read from, for the clients
            Ok(())
        })
    }
    /// Adds an analytic shape of a higher-order GDF file (reference/inputs-meshes §2).
    pub fn analytic_mesh(&mut self, name: &str, shape: &AnalyticShape) -> Result<ObjectId> {
        self.need_changeable()?;
        let mut g = GdfFile {
            higher_order: true,
            geometry: PatchKind::Analytic,
            ulen: shape.reference_length,
            ..GdfFile::default()
        };
        let mut sym = text_value(&shape.symmetry);
        check_value(schema_type("Mesh").item("symmetry").expect("a Mesh has symmetry"), &mut sym)?; // "none", "yz", "xz" or "xz_and_yz"
        let flags = symmetry_flags(&sym);
        g.isx = flags[0];
        g.isy = flags[1];
        g.label = shape.label.clone();
        g.analytic = named_shape(&ShapeSpec {
            shape: shape.shape.clone(),
            dimensions: shape.dimensions.clone(),
            discs: shape.discs.clone(),
            parts: shape.parts.clone(),
            mapping: shape.mapping.clone(),
        })?;
        g.count = g.analytic.len() as i32;
        g.discretization = discretization_rows(&shape.discretization, g.count)?;
        self.new_mesh(name, |m| fill_mesh(m, &g, false))
    }
    /// Adds a mesh from the text of a geometry file (.hbg, reference/inputs-meshes §5), as a client passes the
    /// object it would write; the discretization is the text's.
    pub fn hbg_mesh(&mut self, name: &str, text: &str) -> Result<ObjectId> {
        self.need_changeable()?;
        let o = PatchOptions::default();
        self.new_mesh(name, |m| read_hbg(m, text, "hbg: ", &o, mesh_of_object))
    }
    /// The text of the geometry file (.hbg) of a mesh.
    pub fn mesh_hbg(&self, mesh: ObjectId) -> Result<String> {
        let m = self.obj(mesh)?;
        need_mesh(m)?;
        let _ = mesh_of_object(m)?;
        hbg_text(m)
    }
    /// Writes a panels, flat or spline patch mesh as GDF (and SPL), every number at full precision; the
    /// folder is made; an existing file is ErrorKind::Exists unless overwrite.
    pub fn mesh_write(&self, mesh: ObjectId, gdf_path: &Path, spl_path: Option<&Path>, overwrite: bool) -> Result<()> {
        let m = self.obj(mesh)?;
        need_mesh(m)?;
        let kind = m.mesh_kind();
        let g = path_text(gdf_path)?;
        if is_hbg(g) {
            // a geometry file: one file, every kind but a trim
            if spl_path.is_some() {
                return Err(Error::argument("a geometry file (.hbg) holds the discretization: no SPL file"));
            }
            let _ = mesh_of_object(m)?;
            let text = hbg_text(m)?;
            if !overwrite && file_exists(g) {
                return Err(Error::exists(format!("{g} exists")));
            }
            make_parent_folder(g);
            return write_file(g, text.as_bytes());
        }
        let spl_path = match spl_path {
            Some(p) => Some(path_text(p)?),
            None => None,
        };
        // the texts first: nothing is made for a mesh that cannot be written
        let d = mesh_desc(m)?;
        let mut spl = String::new();
        let gdf;
        if kind == MeshKind::Panels {
            if spl_path.is_some() {
                return Err(Error::argument("a low-order mesh has no SPL file"));
            }
            gdf = gdf_text(&d.lo, &d.label);
        } else if matches!(kind, MeshKind::FlatPatches | MeshKind::SplinePatches) {
            let _ = build_patch_mesh(&d.ho)?;
            gdf = gdf_text_patch(&d.ho, &d.label)?;
            if spl_path.is_some() {
                spl = spl_text(&d.ho, &d.label);
            }
        } else {
            return Err(Error::value(format!(
                "{}a '{}' mesh is kept in the model file, not written as GDF",
                m.prefix(),
                kind.name()
            )));
        }
        // the files: none exists unless overwrite is given, and their folders are made
        for p in [Some(g), spl_path].into_iter().flatten() {
            if !overwrite && file_exists(p) {
                return Err(Error::exists(format!("{p} exists")));
            }
        }
        for p in [Some(g), spl_path].into_iter().flatten() {
            make_parent_folder(p);
        }
        write_file(g, gdf.as_bytes())?;
        if let Some(p) = spl_path {
            write_file(p, spl.as_bytes())?;
        }
        Ok(())
    }
    /// Points and unit normals of the one-based patch element of a mesh at parameters uv in
    /// [-1, 1]^2, mesh coordinates.
    pub fn mesh_patch_points(&self, mesh: ObjectId, element: i32, uv: &[[f64; 2]]) -> Result<PatchPoints> {
        let flat: Vec<f64> = uv.iter().flatten().copied().collect();
        let o = self.obj(mesh)?;
        need_mesh(o)?;
        let m = mesh_of_object(o)?;
        let (p, q) = inspect_patch_points(&m, element, uv.len() as i32, &flat)?;
        let rows = |v: &[f64]| v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Ok(PatchPoints { points: rows(&p), normals: rows(&q) })
    }
    // the engine's mesh of a mesh object of the model
    fn engine_mesh(&self, mesh: ObjectId) -> Result<Mesh> {
        let o = self.obj(mesh)?;
        need_mesh(o)?;
        mesh_of_object(o)
    }
    /// The geometry of a mesh alone, in its own coordinates (no density, gravity or placement): the
    /// volume closed by the mesh and the plane z = 0 (its three estimates and their mean), the centre
    /// of volume, the surface area, the bounds and the waterplane at z = 0; over the whole mesh, its
    /// reflections included.
    pub fn mesh_geometry(&self, mesh: ObjectId) -> Result<MeshGeometry> {
        Ok(mesh_geometry(&self.engine_mesh(mesh)?))
    }
    /// The stored elements of a mesh alone, mesh coordinates, without their reflections: the
    /// one-based element, the vertices, distinct vertices, centroids, unit normals into the fluid
    /// and areas.
    pub fn mesh_elements(&self, mesh: ObjectId) -> Result<MeshElements> {
        Ok(inspect_elements(&self.engine_mesh(mesh)?))
    }
    /// A quadrilateral mesh of a mesh alone for display, mesh coordinates (as `tessellation`; no
    /// role, body -1); divisions per patch in u and v, 1..256, or [0, 0] one quad per span.
    pub fn mesh_tessellation(
        &self,
        mesh: ObjectId,
        divisions: [i32; 2],
        include_reflections: bool,
    ) -> Result<Tessellation> {
        inspect_mesh_tessellation(&self.engine_mesh(mesh)?, divisions, include_reflections)
    }
    /// The geometric quality of a mesh alone (body -1): per stored element its
    /// panel or span size, aspect ratio and flags (orientation, free and non-manifold edges; no wavelength),
    /// the three volumes and the edge counts of its surface closed by the waterline z = 0.
    pub fn mesh_geometry_quality(&self, mesh: ObjectId) -> Result<MeshQuality> {
        Ok(geometry_quality(&self.engine_mesh(mesh)?))
    }
}
