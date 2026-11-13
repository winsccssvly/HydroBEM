//! The geometry of the problem (rule 2 of reference/inputs-rules §1): the bodies, the lids and the
//! fixed dampers, then the roles of the elements of each body's mesh.

use std::collections::BTreeSet;

use super::blank;
use crate::constants::fmt;
use crate::error::{Error, Location, Result, ResultExt, at_body, at_body_element, at_element};
use crate::geometry::{COINCIDENCE_TOL, Vec3, max_of};
use crate::mesh::Range;
use crate::model_codes::{InteriorMethod, Mass, Origin};
use crate::problem::{
    BodyDerived, Problem, Role, all_records, body_label, copies, element_key_of, finite3, model_place, plane_tol,
    records_of,
};
use crate::text::element_text;

const FLT_MAX: f64 = f32::MAX as f64;

// Range validity within a mesh of `count` elements; overlaps are rejected. The document gives
// valid, disjoint ranges (its mesh references are checked where they are set, the parts of a body
// are made disjoint), so an invalid or overlapping range is a bug of the problem's build.
fn check_selection(sel: &[Range], count: i32, what: &str) -> Result<()> {
    let mut s = sel.to_vec();
    for r in &s {
        if r.first < 1 || r.last < r.first {
            return Err(Error::internal(format!("{what}: invalid panel range {}-{}", r.first, r.last)));
        }
        if r.last > count {
            return Err(Error::value(format!(
                "{what}: panel range {}-{} is outside the mesh (elements 1-{count})",
                r.first, r.last
            )));
        }
    }
    s.sort_by_key(|r| r.first);
    for i in 1..s.len() {
        if s[i].first <= s[i - 1].last {
            return Err(Error::internal(format!(
                "{what}: overlapping panel ranges {}-{} and {}-{}",
                s[i - 1].first,
                s[i - 1].last,
                s[i].first,
                s[i].last
            )));
        }
    }
    Ok(())
}

// the element ranges of one surface of a body on the mesh it uses (rule 2)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn claim(
    sel: &[Range],
    role: Role,
    owner: i32,
    what: &str,
    at: &str,
    count: i32,
    bi: i32,
    claimed: &mut [String],
    d: &mut BodyDerived,
) -> Result<()> {
    check_selection(sel, count, &format!("{at} {what}"))?;
    for r in sel {
        for i in r.first..=r.last {
            let k = (i - 1) as usize;
            if !claimed[k].is_empty() {
                return Err(Error::value(format!("{at}: mesh element {i} is taken by both {} and {what}", claimed[k]))
                    .at(at_body_element(bi, i)));
            }
            claimed[k] = what.to_string();
            d.role[k] = role;
            d.owner[k] = owner;
        }
    }
    Ok(())
}

impl Problem {
    // Rule 2: meshes, body fields, element ranges, disjoint surfaces, non-empty
    // wetted surface: the bodies, the lids, the fixed dampers, then the roles of the elements of
    // the mesh each body uses, in this order.
    pub(super) fn rule_geometry(&mut self) -> Result<()> {
        if self.bodies.is_empty() {
            return Err(Error::value("the model has no bodies"));
        }
        self.rule_bodies()?;
        self.rule_lids()?;
        self.rule_fixed_dampers()?;
        self.rule_element_roles()
    }

    // a mesh index of the problem (rule 2)
    fn check_mesh_index(&self, idx: i32, what: &str) -> Result<()> {
        let nmeshes = self.meshes.len() as i32;
        if idx < 0 || idx >= nmeshes {
            return Err(Error::argument(format!("{what}: mesh index {idx} does not exist")));
        }
        Ok(())
    }

    // the items of each body: its name and mesh, its placement and modes, its mass, inertia and
    // matrices with their origins, the meshes of its surfaces, its pressure surfaces and plates
    fn rule_bodies(&self) -> Result<()> {
        let nb = self.bodies.len() as i32;
        let mut names: BTreeSet<String> = BTreeSet::new();
        let opt = self.opt;
        for (bi, b) in self.bodies.iter().enumerate() {
            let here = at_body(bi as i32);
            let at = body_label(b);
            if blank(&b.name) {
                return Err(Error::value("body names must not be blank").at(here));
            }
            if !names.insert(b.name.clone()) {
                return Err(Error::value(format!("duplicate body name '{}'", b.name)).at(here));
            }
            self.check_mesh_index(b.mesh, &at).located(here)?;
            let m = &self.meshes[b.mesh as usize];
            if m.patch != opt.higher_order {
                return Err(Error::value(if opt.higher_order {
                    format!("{at}: options.higher_order = true requires a patch mesh")
                } else {
                    format!("{at}: a patch mesh requires options.higher_order = true")
                })
                .at(here));
            }
            if !finite3(&b.position) {
                return Err(Error::value(format!("{at}: position must be finite")).at(here));
            }
            if !b.yaw.is_finite() {
                return Err(Error::value(format!("{at}: yaw must be finite")).at(here));
            }
            if !finite3(&b.rotation_center) {
                return Err(Error::value(format!("{at}: rotation_center must be finite")).at(here));
            }
            for i in 0..b.modes.len() {
                if b.modes[i] < 1 || b.modes[i] > 6 {
                    return Err(Error::value(format!("{at}: modes must be in 1..6")).at(here));
                }
                if i > 0 && b.modes[i] <= b.modes[i - 1] {
                    return Err(Error::value(format!(
                        "{at}: modes must be unique and in the order surge, sway, heave, roll, pitch, yaw"
                    ))
                    .at(here));
                }
            }
            if b.has_cg && !finite3(&b.cg) {
                return Err(Error::value(format!("{at}: center_of_gravity must be finite")).at(here));
            }
            if let Some(Mass::Given(mass)) = b.mass
                && !(mass.is_finite() && mass >= 0.0)
            {
                return Err(
                    Error::value(format!("{at}: mass must be a finite number >= 0, got {}", fmt(mass))).at(here)
                );
            }
            if b.has_inertia {
                for &v in &b.inertia {
                    if !v.is_finite() || v.abs() > FLT_MAX {
                        return Err(Error::value(format!(
                            "{at}: inertia components must be finite real numbers that fit float32"
                        ))
                        .at(here));
                    }
                }
            }
            if b.has_gyration {
                for &v in &b.gyration {
                    if !v.is_finite() {
                        return Err(
                            Error::value(format!("{at}: radii of gyration must be finite real numbers")).at(here)
                        );
                    }
                }
                // symmetric by the document's check of radii_of_gyration (three values: the diagonal)
                for i in 0..3 {
                    for j in i + 1..3 {
                        if b.gyration[3 * i + j] != b.gyration[3 * j + i] {
                            return Err(Error::internal(format!(
                                "{at}: the radii of gyration must form a symmetric matrix"
                            ))
                            .at(here));
                        }
                    }
                }
            }
            let check_origin = |origin: Origin, pt: &Vec3, what: &str| {
                if origin == Origin::CenterOfGravity && !b.has_cg {
                    return Err(
                        Error::value(format!("{at}: {what}='center_of_gravity' requires center_of_gravity")).at(here)
                    );
                }
                if origin == Origin::Point && !finite3(pt) {
                    return Err(Error::value(format!("{at}: {what} must be finite")).at(here));
                }
                Ok(())
            };
            // the inertia enters the motions of the rigid modes only
            if b.has_inertia && self.opt.rao && !b.modes.is_empty() {
                check_origin(b.inertia_origin, &b.inertia_point, "inertia_origin").located(here)?;
            }
            let check_matrix =
                |a: &Vec<f64>, rows: i32, cols: i32, origin: Origin, pt: &Vec3, what: &str, origin_what: &str| {
                    if a.is_empty() {
                        return Ok(());
                    }
                    // 6 x 6 by the schema, 6 x 6N with the couplings merged in by the build
                    if rows != 6 || (cols != 6 && cols != 6 * nb) {
                        return Err(Error::internal(format!("{at}: {what} must be 6x6 or 6x{}", 6 * nb)).at(here));
                    }
                    for &v in a {
                        if !v.is_finite() || v.abs() > FLT_MAX {
                            return Err(Error::value(format!(
                                "{at}: {what} must contain finite real numbers that fit float32"
                            ))
                            .at(here));
                        }
                    }
                    check_origin(origin, pt, origin_what).located(here)?;
                    Ok(())
                };
            check_matrix(
                &b.stiffness,
                b.stiffness_rows,
                b.stiffness_cols,
                b.stiffness_origin,
                &b.stiffness_point,
                "additional_stiffness",
                "stiffness_origin",
            )
            .located(here)?;
            check_matrix(
                &b.damping,
                b.damping_rows,
                b.damping_cols,
                b.damping_origin,
                &b.damping_point,
                "additional_damping",
                "damping_origin",
            )
            .located(here)?;
            // a 6 x 6N matrix transports its column block j about the origin of body j (motion.rs): those
            // origins are checked as well, of the blocks that are not zero (a zero block transports
            // nothing)
            for k in 0..2 {
                let stiff = k == 0;
                let cols = if stiff { b.stiffness_cols } else { b.damping_cols };
                if cols != 6 * nb || nb == 1 {
                    continue;
                }
                let a = if stiff { &b.stiffness } else { &b.damping };
                let n6 = cols as usize;
                for (j, o) in self.bodies.iter().enumerate() {
                    if (0..6).all(|r| a[r * n6 + 6 * j..r * n6 + 6 * j + 6].iter().all(|&x| x == 0.0)) {
                        continue;
                    }
                    let origin = if stiff { o.stiffness_origin } else { o.damping_origin };
                    if origin == Origin::CenterOfGravity && !o.has_cg {
                        return Err(Error::value(format!(
                            "{at}: the columns of {} in {} are about its {}='center_of_gravity', which it lacks",
                            body_label(o),
                            if stiff { "additional_stiffness" } else { "additional_damping" },
                            if stiff { "stiffness_origin" } else { "damping_origin" }
                        ))
                        .at(here));
                    }
                    if origin == Origin::Point && !finite3(if stiff { &o.stiffness_point } else { &o.damping_point }) {
                        return Err(Error::value(format!(
                            "{}: {} must be finite",
                            body_label(o),
                            if stiff { "stiffness_origin" } else { "damping_origin" }
                        ))
                        .at(here));
                    }
                }
            }
            // external meshes. The document makes the interior surface of elements of the body's mesh
            // (not empty: a trimmed body without a waterline is refused by the trim), of a mesh of
            // its own, or generated (method): the other combinations are bugs of the build
            if b.interior_method == InteriorMethod::PanelRange {
                if b.interior_panels.is_empty() {
                    return Err(Error::internal(format!(
                        "{at}: interior_surface: the elements of the body's mesh are empty"
                    ))
                    .at(here));
                }
                if b.interior_mesh.is_some() {
                    return Err(Error::internal(format!(
                        "{at}: interior_surface: elements of the body's mesh and a mesh of its own"
                    ))
                    .at(here));
                }
            } else if b.interior_method == InteriorMethod::ExternalMesh {
                let Some(im) = b.interior_mesh else {
                    return Err(
                        Error::internal(format!("{at}: interior_surface: a mesh of its own without the mesh")).at(here)
                    );
                };
                if !b.interior_panels.is_empty() {
                    return Err(Error::internal(format!(
                        "{at}: interior_surface: a mesh of its own and elements of the body's mesh"
                    ))
                    .at(here));
                }
                self.check_mesh_index(im as i32, &format!("{at} interior surface")).located(here)?;
                if self.meshes[im].patch != m.patch {
                    return Err(Error::value(format!(
                        "{at}: the interior mesh must be of the same order as the body mesh"
                    ))
                    .at(here));
                }
            } else if !b.interior_panels.is_empty() || b.interior_mesh.is_some() {
                return Err(Error::internal(format!(
                    "{at}: interior_surface: a generated surface (method) with elements or a mesh"
                ))
                .at(here));
            }
            if let Some(cs) = b.control_surface {
                self.check_mesh_index(cs as i32, &format!("{at} control_surface")).located(here)?;
            }
            if let Some(di) = b.dipole_mesh {
                self.check_mesh_index(di as i32, &format!("{at} dipole_mesh")).located(here)?;
                let dm = &self.meshes[di];
                if dm.patch != m.patch {
                    return Err(Error::value(format!(
                        "{at}: dipoles: the mesh of the dipoles must be of the same order as the body mesh"
                    ))
                    .at(here));
                }
                if dm.isx != m.isx || dm.isy != m.isy {
                    return Err(Error::value(format!(
                        "{at}: dipoles: the mesh of the dipoles must have the symmetry of the body mesh"
                    ))
                    .at(here));
                }
            }
            let mut mode_names: BTreeSet<String> = BTreeSet::new();
            for (s, p) in b.pressure.iter().enumerate() {
                let ps = format!("{at} {}", element_text("pressure surface", model_place(p.place, s), &p.name));
                if p.mesh.is_some() == !p.panels.is_empty() {
                    return Err(Error::value(format!("{ps}: give exactly one of panels or mesh")).at(here));
                }
                if let Some(pi) = p.mesh {
                    self.check_mesh_index(pi as i32, &ps).located(here)?;
                    if self.meshes[pi].patch != m.patch {
                        return Err(
                            Error::value(format!("{ps}: the mesh must be of the same order as the body mesh")).at(here)
                        );
                    }
                }
                if p.modes.is_empty() {
                    return Err(Error::value(format!("{ps}: needs at least one PressureMode")).at(here));
                }
                for pm in &p.modes {
                    if blank(&pm.name) {
                        return Err(Error::value(format!("{ps}: pressure mode names must not be blank")).at(here));
                    }
                    if !mode_names.insert(pm.name.clone()) {
                        return Err(Error::value(format!("{at}: duplicate pressure mode name '{}'", pm.name)).at(here));
                    }
                    if !pm.stiffness.is_finite()
                        || !pm.damping.is_finite()
                        || pm.stiffness.abs() > FLT_MAX
                        || pm.damping.abs() > FLT_MAX
                    {
                        return Err(Error::value(format!(
                            "{ps}: mode '{}' stiffness and damping must be finite and fit float32",
                            pm.name
                        ))
                        .at(here));
                    }
                }
            }
            for (s, p) in b.plates.iter().enumerate() {
                let ps = format!("{at} {}", element_text("free-surface plate", model_place(p.place, s), &p.name));
                if p.mesh.is_some() == !p.panels.is_empty() {
                    return Err(Error::value(format!("{ps}: give exactly one of panels or mesh")).at(here));
                }
                if let Some(pi) = p.mesh {
                    self.check_mesh_index(pi as i32, &ps).located(here)?;
                    if self.meshes[pi].patch != m.patch {
                        return Err(
                            Error::value(format!("{ps}: the mesh must be of the same order as the body mesh")).at(here)
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // the lids: their names, damping factor and mesh (of their own, or elements of a body)
    fn rule_lids(&self) -> Result<()> {
        let nb = self.bodies.len() as i32;
        let opt = self.opt;
        let mut lid_names: BTreeSet<String> = BTreeSet::new();
        for (li, l) in self.lids.iter().enumerate() {
            // the rules of a lid, located by its index
            let lid = Location { lid: Some(li), ..Location::default() };
            let mut rules = || -> Result<()> {
                let at = format!("lids/{}", l.name);
                if blank(&l.name) {
                    return Err(Error::value("lid names must not be blank"));
                }
                if !lid_names.insert(l.name.clone()) {
                    return Err(Error::value(format!("duplicate lid name '{}'", l.name)));
                }
                if !(l.damping_factor.is_finite() && l.damping_factor >= 0.0) {
                    return Err(Error::value(format!("{at}: damping_factor must be a finite number >= 0")));
                }
                if l.mesh.is_some() == (l.body >= 0) {
                    return Err(Error::value(format!("{at}: give exactly one of mesh or body with panels")));
                }
                if l.body >= 0 {
                    if l.body >= nb {
                        return Err(Error::argument(format!("{at}: body index {} does not exist", l.body)));
                    }
                    // the elements of a lid placed with a body (a mesh reference, not of a trimmed body)
                    if l.panels.is_empty() {
                        return Err(Error::internal(format!("{at}: a lid placed with a body without elements")));
                    }
                } else {
                    let lm = l.mesh.map_or(-1, |lm| lm as i32); // Some: the rule above
                    self.check_mesh_index(lm, &at)?;
                    if !l.panels.is_empty() {
                        return Err(Error::internal(format!("{at}: a lid of its own mesh with elements of a body")));
                    }
                    if self.meshes[lm as usize].patch != opt.higher_order {
                        return Err(Error::value(format!("{at}: the lid mesh must be of the order of the method")));
                    }
                }
                Ok(())
            };
            rules().located(lid)?;
        }
        Ok(())
    }

    // the fixed dampers: damping factor, mesh, none in the free surface
    fn rule_fixed_dampers(&self) -> Result<()> {
        let opt = self.opt;
        for (k, d) in self.fixed_dampers.iter().enumerate() {
            let at = element_text("fixed_dampers", model_place(d.place, k), &d.name);
            if !(d.epsilon.is_finite() && d.epsilon >= 0.0) {
                return Err(Error::value(format!("{at}: damping_factor must be finite and >= 0")));
            }
            self.check_mesh_index(d.mesh, &at)?;
            let dm = &self.meshes[d.mesh as usize];
            if dm.patch != opt.higher_order {
                return Err(Error::value(format!("{at}: the damper mesh must be of the order of the method")));
            }
            // a sheet in z = 0 coincides with its free-surface image (global coordinates)
            for (r, sh) in dm.shapes.iter().enumerate() {
                if !sh.hull.is_empty() && sh.hull.iter().all(|v| v[2].abs() <= plane_tol(dm, sh)) {
                    return Err(Error::value(format!(
                        "{at}: element {} lies in the free surface z = 0, where a thin sheet coincides with its \
                         free-surface image; a rigid plate there is a free-surface plate",
                        element_key_of(&self.origins, d.mesh, r as i32).0
                    )));
                }
            }
        }
        Ok(())
    }

    // the roles of the elements of the mesh each body uses (BodyDerived): the surfaces' claims,
    // the tanks, the elements in the free surface, the counts, the coordinates and the parity of
    // the generalized modes
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn rule_element_roles(&mut self) -> Result<()> {
        let nb = self.bodies.len() as i32;
        // The element ranges and roles of every element of the mesh each body uses.
        self.derived_ = vec![BodyDerived::default(); self.bodies.len()];
        let mut offset = 0;
        for bi in 0..nb {
            let here = at_body(bi);
            let b = &self.bodies[bi as usize];
            let m = &self.meshes[b.mesh as usize];
            let at = body_label(b);
            // an element named by the rules: the stored element (the numbering of ranges and of the
            // queries), not its place in the engine's mesh
            let origins = &self.origins;
            let stored = |k: usize| element_key_of(origins, b.mesh, k as i32).0;
            let d = &mut self.derived_[bi as usize];
            let count = m.count;
            let n = count as usize;
            d.role = vec![Role::Wet; n];
            d.owner = vec![-1; n];
            let mut claimed: Vec<String> = vec![String::new(); n];
            if b.interior_method == InteriorMethod::PanelRange {
                claim(&b.interior_panels, Role::Interior, -1, "the interior surface", &at, count, bi, &mut claimed, d)
                    .located(here)?;
            }
            claim(&b.dipole_panels, Role::Dipole, -1, "dipoles", &at, count, bi, &mut claimed, d).located(here)?;
            for k in 0..b.dampers.len() {
                let x = &b.dampers[k];
                claim(
                    &x.panels,
                    Role::Dipole,
                    k as i32,
                    &if x.label.is_empty() { element_text("damper", k, &x.name) } else { x.label.clone() },
                    &at,
                    count,
                    bi,
                    &mut claimed,
                    d,
                )
                .located(here)?;
            }
            for s in 0..b.pressure.len() {
                if !b.pressure[s].panels.is_empty() {
                    claim(
                        &b.pressure[s].panels,
                        Role::Pressure,
                        s as i32,
                        &element_text("pressure surface", model_place(b.pressure[s].place, s), &b.pressure[s].name),
                        &at,
                        count,
                        bi,
                        &mut claimed,
                        d,
                    )
                    .located(here)?;
                }
            }
            for s in 0..b.plates.len() {
                if !b.plates[s].panels.is_empty() {
                    claim(
                        &b.plates[s].panels,
                        Role::Plate,
                        s as i32,
                        &element_text("free-surface plate", model_place(b.plates[s].place, s), &b.plates[s].name),
                        &at,
                        count,
                        bi,
                        &mut claimed,
                        d,
                    )
                    .located(here)?;
                }
            }
            for (li, l) in self.lids.iter().enumerate() {
                if l.body == bi {
                    claim(
                        &l.panels,
                        Role::Lid,
                        li as i32,
                        &format!("lids/{}", l.name),
                        &at,
                        count,
                        bi,
                        &mut claimed,
                        d,
                    )
                    .located(here)?;
                }
            }
            // internal tanks: wetted elements become the tank's walls, thin sheets inside keep their role
            d.tank = vec![-1; n];
            d.tank_z = vec![0.0; b.tanks.len()];
            for (ti, t) in b.tanks.iter().enumerate() {
                let w = format!("{at}: {}", element_text("tank", model_place(t.place, ti), &t.name));
                if t.panels.is_empty() {
                    return Err(Error::value(format!("{w} has no panels")).at(here));
                }
                if !(t.density.is_finite() && t.density >= 0.0) {
                    return Err(Error::value(format!("{w}: density_ratio must be finite and >= 0")).at(here));
                }
                check_selection(&t.panels, count, &w).located(here)?;
                // the plates of its free surface on the body mesh: elements of the tank (the build
                // lists them in its panels), their place in the tank's plates the owner
                let mut plate_of: Vec<i32> = vec![-1; n];
                for (s, p) in t.plates.iter().enumerate() {
                    if p.mesh.is_some() {
                        continue;
                    }
                    check_selection(&p.panels, count, &w).located(here)?;
                    for i in records_of(&p.panels) {
                        let inside = t.panels.iter().any(|r| i + 1 >= r.first && i < r.last);
                        if !inside || plate_of[i as usize] >= 0 {
                            return Err(Error::internal(format!(
                                "{w}: element {} of a plate of its free surface is not one of its elements once",
                                stored(i as usize)
                            ))
                            .located(here));
                        }
                        plate_of[i as usize] = s as i32;
                    }
                }
                let mut ztop = f64::NEG_INFINITY;
                let mut wall = false;
                for r in &t.panels {
                    for i in r.first..=r.last {
                        let k = (i - 1) as usize;
                        // an element has one role in a body (the claims of the build): in one tank,
                        // a wall or a sheet of that tank
                        if d.tank[k] >= 0 {
                            return Err(Error::internal(format!("{w}: element {} is in two tanks", stored(k)))
                                .at(at_element(stored(k)))
                                .located(here));
                        }
                        let role = d.role[k];
                        if role != Role::Wet && role != Role::Dipole {
                            return Err(Error::internal(format!("{w}: element {} is also {}", stored(k), claimed[k]))
                                .at(at_element(stored(k)))
                                .located(here));
                        }
                        d.tank[k] = ti as i32;
                        if role == Role::Wet {
                            // a wall, or a plate of the tank's free surface: an element of the
                            // tank's boundary, its plate the owner (theory §10.2)
                            d.role[k] = Role::Tank;
                            d.owner[k] = plate_of[k];
                            wall |= plate_of[k] < 0;
                        } else if plate_of[k] >= 0 {
                            return Err(Error::internal(format!("{w}: element {} is a sheet and a plate", stored(k)))
                                .at(at_element(stored(k)))
                                .located(here));
                        }
                        for v in &m.shapes[k].hull {
                            ztop = max_of(ztop, v[2]);
                        }
                    }
                }
                // the tank's mesh is its walls (its sheets are parts of their own), cut by the trim at
                // the tank's free surface with its sheets
                if !wall {
                    return Err(Error::internal(format!("{w} has no wall elements (only thin sheets)")).at(here));
                }
                match t.surface_level {
                    None => d.tank_z[ti] = ztop,
                    Some(level) => {
                        if !level.is_finite() {
                            return Err(Error::value(format!("{w}: surface_level must be finite")).at(here));
                        }
                        if ztop > level + COINCIDENCE_TOL * max_of(1.0, ztop.abs()) {
                            return Err(
                                Error::value(format!("{w}: a tank element lies above its surface level")).at(here)
                            );
                        }
                        d.tank_z[ti] = level;
                    }
                }
                // the plates of its free surface lie at its surface level (body-local z)
                for (s, p) in t.plates.iter().enumerate() {
                    let ps = format!("{w} {}", element_text("free-surface plate", model_place(p.place, s), &p.name));
                    let (pm, recs) = if let Some(pi) = p.mesh {
                        let pm = &self.meshes[pi];
                        (pm, all_records(pm))
                    } else {
                        (m, records_of(&p.panels))
                    };
                    if recs.is_empty() {
                        return Err(Error::value(format!("{ps} has no elements")).at(here));
                    }
                    for rec in recs {
                        let sh = &pm.shapes[rec as usize];
                        if sh.hull.iter().any(|v| (v[2] - d.tank_z[ti]).abs() > plane_tol(pm, sh)) {
                            return Err(Error::value(format!(
                                "{ps}: an element does not lie in the tank's free surface z = {} (body \
                                 coordinates; a plate of a tank lies at its surface level)",
                                d.tank_z[ti]
                            ))
                            .at(here));
                        }
                    }
                }
            }
            d.symmetry = copies(m);
            // a wetted element wholly in the free surface (a lid or an interior surface given as
            // part of the hull): its source and its image coincide, the system is singular
            for k in 0..d.role.len() {
                if d.role[k] == Role::Wet {
                    let sh = &m.shapes[k];
                    let tol = plane_tol(m, sh);
                    if !sh.hull.is_empty() && sh.hull.iter().all(|v| (v[2] + b.position[2]).abs() <= tol) {
                        return Err(Error::value(format!(
                            "{at}: element {} lies in the free surface z = 0; give it as the interior surface, a lid or a free-surface plate, or leave it out of the body's elements",
                            stored(k)
                        ))
                        .at(at_element(stored(k)))
                        .located(here));
                    }
                }
                // a thin sheet in z = 0 coincides with its free-surface image (its hypersingular kernel
                // and that of the image cancel nowhere) and has water on one side only
                if d.role[k] == Role::Dipole && d.tank[k] < 0 {
                    let sh = &m.shapes[k];
                    if !sh.hull.is_empty() && sh.hull.iter().all(|v| (v[2] + b.position[2]).abs() <= plane_tol(m, sh)) {
                        return Err(Error::value(format!(
                            "{at}: element {}, a thin sheet (a dipole or a damper), lies in the free surface z = 0, \
                             where it coincides with its free-surface image; a rigid plate there is a free-surface \
                             plate",
                            stored(k)
                        ))
                        .at(at_element(stored(k)))
                        .located(here));
                    }
                }
            }
            if let Some(di) = b.dipole_mesh {
                let dm = &self.meshes[di];
                for (k, sh) in dm.shapes.iter().enumerate() {
                    if !sh.hull.is_empty() && sh.hull.iter().all(|v| (v[2] + b.position[2]).abs() <= plane_tol(dm, sh))
                    {
                        return Err(Error::value(format!(
                            "{at}: element {} of the mesh of its dipoles lies in the free surface z = 0, where a thin \
                             sheet coincides with its free-surface image; a rigid plate there is a free-surface plate",
                            element_key_of(origins, di as i32, k as i32).0
                        ))
                        .located(here));
                    }
                }
            }
            for k in 0..d.role.len() {
                let r = d.role[k];
                if r == Role::Wet {
                    d.wet += 1;
                }
                if r == Role::Dipole && d.tank[k] < 0 {
                    d.sheet += 1; // a sheet in a tank belongs to the tank problem
                }
                if r == Role::Interior {
                    d.interior += 1;
                }
                if r == Role::Plate {
                    d.plate += 1;
                }
            }
            for p in &b.plates {
                if let Some(pi) = p.mesh {
                    d.plate += self.meshes[pi].count;
                }
            }
            if let Some(di) = b.dipole_mesh {
                d.sheet += self.meshes[di].count;
            }
            if b.interior_method == InteriorMethod::ExternalMesh {
                d.interior = self.meshes
                    [b.interior_mesh.expect("an external interior surface has its mesh (rule_bodies)")]
                .count;
            }
            if b.interior_method == InteriorMethod::Projected
                || b.interior_method == InteriorMethod::Waterline
                || b.interior_method == InteriorMethod::Graded
            {
                d.interior = -1;
            }
            for p in &b.pressure {
                d.pressure_modes += p.modes.len() as i32;
            }
            d.generalized = b.generalized.len() as i32;
            d.coordinate_offset = offset;
            offset += 6 + d.generalized + d.pressure_modes;
            for g in &b.generalized {
                for pc in &g.pieces {
                    if !pc.panels.is_empty() {
                        check_selection(&pc.panels, count, &format!("{at} generalized mode '{}'", g.name))
                            .located(here)?;
                    }
                }
                // the declared parity under the mesh reflections, at every element centre
                const PAR: [[i32; 2]; 6] = [[-1, 1], [1, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]; // (sx, sy) signs
                for r in 0..count {
                    let sh = &m.shapes[r as usize];
                    let (c, nrm) = (sh.centre, sh.normal);
                    let u = g.displacement(&c, r + 1);
                    let v = nrm[0] * u[0] + nrm[1] * u[1] + nrm[2] * u[2];
                    let mut scale: f64 = 0.0;
                    for i in 0..3 {
                        scale = max_of(scale, u[i].abs());
                    }
                    for axis in 0..2 {
                        if (if axis == 0 { m.isx } else { m.isy }) == 0 {
                            continue;
                        }
                        let (mut rc, mut rn) = (c, nrm);
                        rc[axis] = -rc[axis];
                        rn[axis] = -rn[axis];
                        let e = g.displacement(&rc, r + 1);
                        let ve = rn[0] * e[0] + rn[1] * e[1] + rn[2] * e[2];
                        let sign = PAR[(g.parity - 1) as usize][axis] as f64;
                        if (ve - sign * v).abs() > 1e-9 * max_of(scale, 1.0) * (1.0 + v.abs()) {
                            return Err(Error::value(format!(
                                    "{at}: generalized mode '{}' does not have the parity of its parity mode under the reflection of the mesh about {} (element {})",
                                    g.name,
                                    if axis == 0 { "x = 0" } else { "y = 0" },
                                    stored(r as usize)
                                )).at(at_element(stored(r as usize))).located(here));
                        }
                    }
                }
            }
            // a body has a wetted surface, dipoles or free-surface plates (the document), its
            // elements not taken by another role, and a trim that leaves none is refused by the trim
            if d.wet + d.sheet + d.plate == 0 {
                return Err(Error::internal(format!(
                    "{at}: the wetted surface is empty after removing the selected panels"
                ))
                .at(here));
            }
            if !b.body_points.is_empty() && d.wet + d.plate == 0 {
                return Err(Error::value(format!(
                    "{at}: body_points need wetted panels or free-surface plates (the surface has thin sheets only)"
                ))
                .at(here));
            }
        }
        Ok(())
    }
}
