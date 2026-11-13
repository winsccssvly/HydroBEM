//! The trimmed meshes of a model (model.rs): the internal trim of a mesh (`Model::trim_mesh`, which
//! the build of a body with a trim uses too) and the surface of one role of a body as the engine
//! uses it, added to the meshes of the model (`Model::trimmed_body_mesh`, with the generated
//! interior surface of a low-order body).

use crate::document::{ObjectId, symmetry_value};
use crate::document_build::{
    Part, body_mesh as build_body_mesh, body_mesh_name, keep_records, own_generated_plates, placed_fixed_dampers,
};
use crate::document_enabled::{Scope, scoped_document};
use crate::document_mesh::mesh_desc;
use crate::error::{Error, Result};
use crate::geometry::Vec3;
use crate::interior::interior_panels;
use crate::mesh::build_mesh;
use crate::mesh_desc::{MeshKind, TrimDesc};
use crate::model::Model;
use crate::model_meshes::{
    check_body_mesh, int_array_value, int_value, need_mesh, real_array_value, real_value, text_value,
};
use crate::model_types::Trim;
use crate::preprocess::{trim_higher_order, trim_low_order};
use crate::text::py_str_repr;

// the corners of quadrilaterals as rows of 12 numbers
fn flat_panels(panels: &[[Vec3; 4]]) -> Vec<f64> {
    let mut v = Vec::with_capacity(12 * panels.len());
    for p in panels {
        v.extend(p.iter().flat_map(|x| x.0));
    }
    v
}

impl Model {
    /// Adds the trimmed mesh of a mesh: low order a panels mesh, higher order a
    /// trimmed_patches mesh (with the interior surface, of a copy of its source with the interior
    /// patches); its source_elements give the element each element comes from. The trim of a body
    /// (`trim` of a Body, document_build.rs) and `body_trimmed_mesh` use it.
    pub(crate) fn trim_mesh(&mut self, mesh: ObjectId, name: &str, trim: &Trim) -> Result<ObjectId> {
        self.trim_mesh_copies(mesh, name, trim).map(|(id, _, _)| id)
    }
    /// `trim_mesh`, with per element of the trimmed mesh the copy (sx, sy) of its source element
    /// it was cut from (the elements the trim generated: (1, 1)) and the generated plate of
    /// `trim.plates` it is part of (-1 none).
    pub(crate) fn trim_mesh_copies(
        &mut self,
        mesh: ObjectId,
        name: &str,
        trim: &Trim,
    ) -> Result<(ObjectId, Vec<[i32; 2]>, Vec<i32>)> {
        self.changeable(mesh)?;
        let s = trim;
        let src = self.obj(mesh)?;
        need_mesh(src)?;
        let n = name;
        let kind = src.mesh_kind();
        if kind == MeshKind::TrimmedPatches {
            return Err(Error::value(format!("{}a trimmed mesh is not trimmed again", src.prefix())));
        }
        // the name first: nothing is made for a trim that cannot be added
        if self.doc.root().child("meshes", n).is_some() {
            return Err(Error::value(format!("duplicate mesh name {}", py_str_repr(n))));
        }
        let mut t = TrimDesc {
            elevation: s.elevation,
            pitch: s.pitch,
            roll: s.roll,
            interior_surface: s.interior_surface as i32,
            plates: s.plates.clone(),
            ..TrimDesc::default()
        };
        let d = mesh_desc(src)?;
        let src_name = src.name().to_string();
        let src_length = src.get("reference_length").real();
        let src_label = src.get("label").text().to_string();
        if kind == MeshKind::Panels {
            if s.interior_surface {
                return Err(Error::value(format!(
                    "{}the interior free surface is generated for patch meshes only (low order: interior_surface 'graded' or 'waterline')",
                    src.prefix()
                )));
            }
            let m = build_mesh(&d.lo)?;
            // the elements of each tank cut at its free surface, the others at z = 0 (NaN)
            let mut cut = vec![f64::NAN; m.count.max(0) as usize];
            for (records, level) in &s.tanks {
                if !level.is_finite() {
                    return Err(Error::value("trim: the tank levels must be finite"));
                }
                for &i in records {
                    if i < 1 || i > m.count {
                        return Err(Error::value(format!(
                            "trim: tank element {} does not exist ({} elements)",
                            i, m.count
                        )));
                    }
                    if !cut[(i - 1) as usize].is_nan() {
                        return Err(Error::value(format!("trim: element {i} is in two tanks")));
                    }
                    cut[(i - 1) as usize] = *level;
                }
            }
            if !s.tanks.is_empty() {
                t.record_cut = Some(cut);
            }
            let r = trim_low_order(&m, &t)?;
            let v = flat_panels(&r.panels);
            let np = r.panels.len() as i64;
            let srcrec: Vec<i64> = r.source.iter().map(|&x| x as i64).collect();
            let id = self.new_mesh(n, |o| {
                o.set("kind", text_value(MeshKind::Panels.name()))?;
                o.set("reference_length", real_value(src_length))?;
                o.set("symmetry", symmetry_value(r.symmetry[0], r.symmetry[1]))?;
                o.set("label", text_value(&src_label))?;
                o.set("vertices", real_array_value(vec![np, 4, 3], v))?;
                o.set("source_elements", int_array_value(vec![np], srcrec))?;
                Ok(())
            })?;
            return Ok((id, r.copy, r.plate));
        }
        if !s.tanks.is_empty() {
            return Err(Error::value(format!(
                "{}the tank free surfaces of a trim are for low-order meshes",
                src.prefix()
            )));
        }
        let r = trim_higher_order(&d.ho, &t)?;
        let copies: Vec<[i32; 2]> = r.records.iter().map(|x| if x.patch < 0 { [1, 1] } else { [x.sx, x.sy] }).collect();
        let mut plates = vec![-1; r.records.len() - r.interior.len()];
        plates.extend_from_slice(&r.interior_plate);
        let nr = r.records.len() as i64;
        let ni = r.interior.len() as i64;
        let mut source = src_name;
        let mut source_copy: Option<ObjectId> = None; // removed again when the trim fails after it was made
        let made = (|| -> Result<ObjectId> {
            if ni != 0 {
                // the interior free-surface patches appended to a copy of the source
                let cn = self.unique_mesh_name(&format!("{n}_source"));
                let copy = self.new_mesh(&cn, |o| {
                    let id = o.id();
                    o.doc_mut().copy_items(id, mesh)
                })?;
                source_copy = Some(copy);
                let mut copy = self.doc.obj_mut(copy);
                let q = copy.get("discretization");
                let mut disc = q.ints().to_vec();
                for x in &r.interior_disc {
                    disc.extend_from_slice(&[
                        x.nu as i64,
                        x.nv as i64,
                        x.ku as i64,
                        x.kv as i64,
                        x.iquo as i64,
                        x.iqui as i64,
                    ]);
                }
                if kind == MeshKind::FlatPatches {
                    let mut v = copy.get("vertices").reals().to_vec();
                    v.extend(flat_panels(&r.interior));
                    let nv = (v.len() / 12) as i64;
                    copy.set("vertices", real_array_value(vec![nv, 4, 3], v))?;
                } else if kind == MeshKind::AnalyticPatches {
                    for p in &r.interior {
                        let mut a = copy.add("patches", "AnalyticPatch", "", -1)?;
                        a.set("map", text_value("bilinear"))?;
                        let v = flat_panels(std::slice::from_ref(p));
                        a.set("vertices", real_array_value(vec![4, 3], v))?;
                    }
                } else {
                    // a bilinear B-spline patch of order 2 (four control points)
                    for p in &r.interior {
                        let mut sp = copy.add("patches", "SplinePatch", "", -1)?;
                        sp.set("u_order", int_value(2))?;
                        sp.set("v_order", int_value(2))?;
                        sp.set("u_knots", real_array_value(vec![4], vec![0.0, 0.0, 1.0, 1.0]))?;
                        sp.set("v_knots", real_array_value(vec![4], vec![0.0, 0.0, 1.0, 1.0]))?;
                        let mut c = Vec::new();
                        for k in [0, 1, 3, 2] {
                            c.extend_from_slice(&p[k].0);
                        }
                        sp.set("coefficients", real_array_value(vec![2, 2, 3], c))?;
                    }
                }
                let nd = (disc.len() / 6) as i64;
                copy.set("discretization", int_array_value(vec![nd, 6], disc))?;
                source = copy.as_obj().name().to_string();
            }
            let (mut rec, mut disc, mut rng) = (Vec::new(), Vec::new(), Vec::new());
            let srcrec: Vec<i64> = r.source.iter().map(|&x| x as i64).collect();
            for x in &r.records {
                // the source patch one-based
                rec.extend_from_slice(&[x.patch as i64 + 1, x.sx as i64, x.sy as i64, x.cut as i64]);
                rng.extend_from_slice(&x.range);
                let q = &x.discretization;
                disc.extend_from_slice(&[
                    q.nu as i64,
                    q.nv as i64,
                    q.ku as i64,
                    q.kv as i64,
                    q.iquo as i64,
                    q.iqui as i64,
                ]);
            }
            self.new_mesh(n, |o| {
                o.set("kind", text_value(MeshKind::TrimmedPatches.name()))?;
                o.set("source", text_value(&source))?;
                o.set("trim", real_array_value(vec![3], vec![s.elevation, s.pitch, s.roll]))?;
                o.set("symmetry", symmetry_value(r.symmetry[0], r.symmetry[1]))?;
                o.set("trim_elements", int_array_value(vec![nr, 4], rec))?;
                o.set("parameter_ranges", real_array_value(vec![nr, 4], rng))?;
                o.set("discretization", int_array_value(vec![nr, 6], disc))?;
                o.set("source_elements", int_array_value(vec![nr], srcrec))?;
                Ok(())
            })
        })();
        let r = match made {
            Ok(m) => Ok(m),
            Err(e) => match source_copy {
                Some(c) => {
                    let root = self.doc.root_id();
                    self.doc.remove_child(root, c).and(Err(e))
                }
                None => Err(e),
            },
        };
        self.sync();
        r.map(|id| (id, copies, plates))
    }
    /// Adds the surface of one role of a body as the engine uses it (after its trim) to the meshes of
    /// the model: role "wetted", "dipole", "dipole_damper", "tank", "pressure_surface",
    /// "free_surface_plate" or "interior_surface" (the elements of the body's mesh, or the interior
    /// surface the trim of a patch mesh generates); low order a panels mesh, higher order a
    /// trimmed_patches mesh, its source_elements the stored elements of the body's mesh they come
    /// from.
    pub fn body_trimmed_mesh(&mut self, body: ObjectId, name: &str, role: &str) -> Result<ObjectId> {
        self.changeable(body)?;
        let b = self.own_body(body)?;
        // a generated interior surface of the low order: the surface the solve generates
        let generated = role == "interior_surface"
            && !self.doc.root().child("options", "").is_some_and(|o| o.get("higher_order").boolean())
            && matches!(b.get("interior_surface").text(), "waterline" | "graded" | "projected");
        if generated {
            return self.generated_interior_mesh(body, name);
        }
        let b = self.own_body(body)?;
        let at = b.prefix();
        // the parts of the role (the sheets inside the tanks with the dipoles and dampers)
        let of_role: fn(&Part) -> bool = match role {
            "wetted" => |p| matches!(p, Part::Wet),
            "dipole" => |p| matches!(p, Part::Dipole(_) | Part::TankDipole(..)),
            "dipole_damper" => |p| matches!(p, Part::Damper(_) | Part::TankDamper(..)),
            "tank" => |p| matches!(p, Part::Tank(_)),
            "pressure_surface" => |p| matches!(p, Part::Pressure(_)),
            "free_surface_plate" => |p| matches!(p, Part::Plate(_) | Part::TankPlate(..)),
            "interior_surface" => |p| matches!(p, Part::Interior),
            _ => {
                return Err(Error::value(format!(
                    "role must be one of wetted, dipole, dipole_damper, tank, pressure_surface, free_surface_plate, interior_surface, got {}",
                    py_str_repr(role)
                )));
            }
        };
        if self.doc.root().child("meshes", name).is_some() {
            return Err(Error::value(format!("duplicate mesh name {}", py_str_repr(name))));
        }
        check_body_mesh(b)?;
        // of the enabled objects, the body itself taken as enabled
        let enabled = scoped_document(&self.doc, Scope::BodyMesh(body))?;
        let bm = {
            let eb = enabled.doc.obj(body);
            let placed = placed_fixed_dampers(&enabled.doc)?;
            build_body_mesh(eb, &placed[eb.index()])?
        };
        let own = bm.own.clone();
        // the engine elements of the role (and the generated interior surface)
        let mut recs: Vec<i32> = Vec::new();
        for (p, r) in &bm.parts {
            if of_role(p) {
                recs.extend(bm.records(r));
            }
        }
        if role == "interior_surface" {
            for r in bm.generated() {
                recs.extend(r.first..=r.last);
            }
        }
        if role == "free_surface_plate" {
            // the plates the trim generated (or the mesh of a body of generated plates only)
            recs.extend((0..bm.plate.len()).filter(|&k| bm.plate[k] >= 0).map(|k| k as i32 + 1));
        }
        recs.sort();
        if recs.is_empty() {
            return Err(Error::value(format!("{at}the body has no elements of the role {}", py_str_repr(role))));
        }
        let keep: Vec<bool> = {
            let mut k = vec![false; bm.source.len()];
            for &r in &recs {
                k[(r - 1) as usize] = true;
            }
            k
        };
        let source: Vec<i64> = recs.iter().map(|&r| bm.source[(r - 1) as usize] as i64).collect();
        let mut d = bm.desc.clone();
        keep_records(&mut d, &keep);
        let np = recs.len() as i64;
        let src = self.doc.root().child("meshes", &own).expect("the mesh of a body");
        let (length, label) = (src.get("reference_length").real(), src.get("label").text().to_string());
        if d.low_order {
            let sym = symmetry_value(d.lo.symmetry_x, d.lo.symmetry_y);
            return self.new_mesh(name, |o| {
                o.set("kind", text_value(MeshKind::Panels.name()))?;
                o.set("reference_length", real_value(length))?;
                o.set("symmetry", sym)?;
                o.set("label", text_value(&label))?;
                o.set("vertices", real_array_value(vec![np, 4, 3], d.lo.vertices.clone()))?;
                o.set("source_elements", int_array_value(vec![np], source))?;
                Ok(())
            });
        }
        if !bm.trimmed {
            return Err(Error::value(format!(
                "{at}the trimmed surface of a role: a patch mesh of a body without trim is the body's mesh itself (give its elements by a mesh reference)"
            )));
        }
        // a trimmed patch mesh: the source as the trim of the body made it (with the generated interior
        // and plate patches appended to a copy)
        let t = b.get("trim").reals().to_vec();
        let interior = !bm.generated().is_empty();
        let plates = own_generated_plates(b)?.iter().map(|g| g.1).collect();
        let mesh = src.id();
        let made = self.trim_mesh(
            mesh,
            name,
            &Trim { elevation: t[0], pitch: t[1], roll: t[2], interior_surface: interior, tanks: Vec::new(), plates },
        )?;
        let (mut rec, mut disc, mut rng) = (Vec::new(), Vec::new(), Vec::new());
        for x in &d.ho.trim {
            rec.extend_from_slice(&[x.patch as i64 + 1, x.sx as i64, x.sy as i64, x.cut as i64]); // one-based patch
            rng.extend_from_slice(&x.range);
            let q = &x.discretization;
            disc.extend_from_slice(&[q.nu as i64, q.nv as i64, q.ku as i64, q.kv as i64, q.iquo as i64, q.iqui as i64]);
        }
        let mut srcs: Vec<i64> = Vec::new();
        for x in &d.ho.trim {
            srcs.push(x.patch as i64 + 1); // the source patch; an interior patch beyond the stored elements
        }
        let r = (|| -> Result<()> {
            let mut o = self.doc.obj_mut(made);
            o.set("trim_elements", int_array_value(vec![np, 4], rec))?;
            o.set("parameter_ranges", real_array_value(vec![np, 4], rng))?;
            o.set("discretization", int_array_value(vec![np, 6], disc))?;
            o.set("source_elements", int_array_value(vec![np], srcs))?;
            Ok(())
        })();
        if r.is_err() {
            // nothing is left of a failed trim: the trimmed mesh and the copy of its source it made
            let root = self.doc.root_id();
            let source = self.doc.obj(made).get("source").text().to_string();
            self.doc.remove_child(root, made)?;
            if source != own {
                if let Some(copy) = self.doc.root().child("meshes", &source).map(|c| c.id()) {
                    self.doc.remove_child(root, copy)?;
                }
            }
        }
        self.sync();
        r.map(|_| made)
    }
    // the generated interior surface of a low-order body as the solve makes it (after the model
    // rules), body coordinates in the stored symmetry form, as a panels mesh of the model; its
    // source_elements 0 (elements the engine made)
    fn generated_interior_mesh(&mut self, body: ObjectId, name: &str) -> Result<ObjectId> {
        if self.doc.root().child("meshes", name).is_some() {
            return Err(Error::value(format!("duplicate mesh name {}", py_str_repr(name))));
        }
        let own = body_mesh_name(self.own_body(body)?);
        let ip = self.body_query(body, |p, bi| interior_panels(p, bi))?;
        let src = own.and_then(|n| self.doc.root().child("meshes", &n));
        let (length, label) =
            src.map_or((1.0, String::new()), |s| (s.get("reference_length").real(), s.get("label").text().to_string()));
        let np = ip.panels.len() as i64;
        let vertices: Vec<f64> = ip.panels.iter().flat_map(|p| p.raw.iter().flat_map(|x| x.0)).collect();
        self.new_mesh(name, |o| {
            o.set("kind", text_value(MeshKind::Panels.name()))?;
            o.set("reference_length", real_value(length))?;
            o.set("symmetry", symmetry_value(ip.isx, ip.isy))?;
            o.set("label", text_value(&label))?;
            o.set("vertices", real_array_value(vec![np, 4, 3], vertices))?;
            o.set("source_elements", int_array_value(vec![np], vec![0; np as usize]))?;
            Ok(())
        })
    }
}
