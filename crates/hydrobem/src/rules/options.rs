//! The options and their combinations (rule 1 of reference/inputs-rules §1, with the potential
//! QTF's free surface and the footprint of the control surfaces), something to solve (rule 4),
//! the motions (rule 5) and the integration and formulation the surfaces need (rule 8).

use std::collections::BTreeSet;

use super::blank;
use crate::constants::fmt;
use crate::error::{Error, Result, at_body};
use crate::model_codes::{Domain, Excitation, Formulation, InteriorMethod};
use crate::plate_geometry::plane_vertices;
use crate::polygon::{convex_hull, inside_convex};
use crate::problem::{Problem, body_label, finite3, plane_tol};
use crate::text::element_text;

impl Problem {
    // The rules of the potential QTF's free surface (second_order.free_surface_*, the quadrature
    // zone, the outer circle; reference/inputs-orders §3): the given mesh, and with the potential
    // QTF the ranges of the items, the method and the domain its zones are made in. A part of
    // rule 1; also the rules of the free-surface queries (check_free_surface_rules).
    pub(super) fn rule_free_surface(&self) -> Result<()> {
        let opt = &self.opt;
        if let Some(fm) = opt.free_surface_mesh {
            // the free surface of the potential QTF from a mesh: its panels in z = 0 set R and the
            // points instead of the cells
            if !opt.potential_qtf.any() {
                return Err(Error::value(
                    "second_order.free_surface_method = 'mesh' needs qtf_potential_direct or qtf_potential_indirect (the mesh is used only by the potential QTF)",
                ));
            }
            if opt.potential_qtf_radius.is_some() || opt.potential_qtf_panel_size.is_some() {
                return Err(Error::internal("second_order: a given mesh with the items of the generated zone"));
            }
            let fm = &self.meshes[fm];
            if fm.patch {
                return Err(Error::value("second_order: the mesh must be of panels"));
            }
            if fm.panels.iter().any(|p| p.raw.iter().any(|x| x[2].abs() > 1e-6 * fm.reference_length.max(1.0))) {
                return Err(Error::value("second_order: every vertex must lie in z = 0"));
            }
        }
        if opt.potential_qtf.any() {
            // the ranges of the items of second_order.free_surface_*, checked where they are set (the schema):
            // a value out of them here is a bug of the problem's build
            for (name, v) in [("inner_radius", opt.potential_qtf_radius), ("panel_size", opt.potential_qtf_panel_size)]
            {
                if let Some(v) = v
                    && !(v.is_finite() && v > 0.0)
                {
                    return Err(Error::internal(format!(
                        "second_order.free_surface_{name} must be positive or auto, got {v}"
                    )));
                }
            }
            let st = opt.potential_qtf_annulus_step;
            if opt.potential_qtf_annuli < 0 || st.is_some_and(|st| !(st.is_finite() && st > 0.0)) {
                return Err(Error::internal(
                    "second_order.quadrature_annuli must be >= 0 and second_order.quadrature_radius_step positive or auto",
                ));
            }
            if opt.potential_qtf_radial_nodes < 1
                || opt.potential_qtf_azimuthal_nodes.is_some_and(|n| n < 1)
                || opt.potential_qtf_segments.is_some_and(|n| n < 1)
                || opt.potential_qtf_order.is_some_and(|n| !(1..=160).contains(&n))
            {
                return Err(Error::internal(
                    "second_order: the nodes and segments must be >= 1 (or auto), the expansion order 1..160 (or auto)",
                ));
            }
            if opt.higher_order {
                return Err(Error::value(
                    "the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect) is available in the low order (not yet in the higher order)",
                ));
            }
            // the open sea and one wall (the open-sea problem of the bodies and their mirror
            // images, theory §12.14); not two walls or a channel, whose images are not one
            // mirror image, nor lids or fixed dampers on the free surface
            if !matches!(opt.domain, Domain::OpenSea | Domain::XWall | Domain::YWall) {
                return Err(Error::value(
                    "the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect) is available in the open sea and with one wall (x_wall, y_wall), not with orthogonal_walls or a channel",
                ));
            }
            if !self.lids.is_empty() || !self.fixed_dampers.is_empty() {
                return Err(Error::value(
                    "the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect) is not available with damping lids or fixed dampers",
                ));
            }
            if opt.domain != Domain::OpenSea && (opt.free_surface_mesh.is_some() || !opt.potential_qtf_panels) {
                return Err(Error::value(
                    "second_order: with a wall the free surface is the generated panels (free_surface_method = 'panels')",
                ));
            }
        }
        Ok(())
    }

    // Rule 1: the options and their combinations (the types and choices of the items are checked
    // where they are set; these checks also cover a problem built in the engine).
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    pub(super) fn rule_options(&mut self) -> Result<()> {
        let opt = &self.opt;
        let bodies = &self.bodies;
        if opt.drift_angles < -1 || opt.drift_angles > 1 << 16 {
            return Err(Error::value("second_order.drift_momentum_conservation_angles must be in 1..65536 or 'auto'"));
        }
        if opt.drift_angles != 0 && !opt.diffraction {
            return Err(Error::value("second_order.drift_momentum_conservation requires the diffraction solution"));
        }
        if opt.pressure_drift && !opt.diffraction {
            return Err(Error::value("second_order.drift_pressure_integration requires the diffraction solution"));
        }
        if (opt.pressure_qtf.any() || opt.potential_qtf.any() || opt.potential_qtf_direct) && !opt.diffraction {
            return Err(Error::value(
                "the QTFs (second_order.qtf_quadratic_..., qtf_potential_...) require the diffraction solution",
            ));
        }
        if opt.pressure_qtf.any() && !opt.pressure_drift {
            return Err(Error::internal("the quadratic QTFs without the pressure drift"));
        }
        self.rule_free_surface()?;
        if opt.potential_qtf.any() {
            for (bi, b) in bodies.iter().enumerate() {
                if !b.pressure.is_empty() || !b.tanks.is_empty() {
                    return Err(Error::value(format!(
                        "{}: the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect) is available for bodies without pressure surfaces or tanks",
                        body_label(b)
                    ))
                    .at(at_body(bi as i32)));
                }
                // the sub-problems of the potential QTF need the body velocities, which the low
                // order does not give with thin sheets, and the jump of a damper is not in them
                if !b.dipole_panels.is_empty() || b.dipole_mesh.is_some() || !b.dampers.is_empty() {
                    return Err(Error::value(format!(
                        "{}: the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect) is not available with thin sheets (dipoles) or dipole dampers",
                        body_label(b)
                    ))
                    .at(at_body(bi as i32)));
                }
                if (!b.modes.is_empty() || !b.generalized.is_empty()) && !opt.rao {
                    return Err(Error::value(format!(
                        "{}: the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect) of a body with modes or generalized modes needs first_order.motion = true (its motions in the forcing); give modes = [] for a fixed body",
                        body_label(b)
                    ))
                    .at(at_body(bi as i32)));
                }
            }
        }
        if opt.control_surface_qtf && (!opt.control_surface_drift || !opt.pressure_qtf.any()) {
            return Err(Error::internal("the control-surface QTFs without their drift or the pressure QTFs"));
        }
        for (bi, b) in bodies.iter().enumerate() {
            if opt.pressure_drift && !b.tanks.is_empty() && opt.higher_order {
                return Err(Error::value(format!(
                    "{}: second_order.drift_pressure_integration with internal tanks is available in the low order only",
                    body_label(b)
                ))
                .at(at_body(bi as i32)));
            }
        }
        for (bi, b) in bodies.iter().enumerate() {
            // the generalized modes in the pressure drift: the low order, the wetted surface
            // integrated directly (not the closed forms of pressure surfaces), no tank walls (their
            // drift moves them rigidly)
            if opt.pressure_drift && !b.generalized.is_empty() && opt.higher_order {
                return Err(Error::value(format!(
                    "{}: second_order.drift_pressure_integration with generalized modes is available in the low order only",
                    body_label(b)
                ))
                .at(at_body(bi as i32)));
            }
            if opt.pressure_drift && !b.generalized.is_empty() && (!b.pressure.is_empty() || !b.tanks.is_empty()) {
                return Err(Error::value(format!(
                    "{}: second_order.drift_pressure_integration with generalized modes is not available with pressure surfaces or tanks",
                    body_label(b)
                ))
                .at(at_body(bi as i32)));
            }
            // the second-order loads take the exact surface of the displaced body: a mode whose
            // pieces tear the wetted surface open has none
            let second = opt.pressure_drift || opt.potential_qtf.any() || opt.potential_qtf_direct;
            if second && !b.generalized.is_empty() && !opt.higher_order {
                if let Some((g, x, ra, rb)) = self.torn_mode(bi) {
                    return Err(Error::value(format!(
                        "{}: generalized mode '{}' jumps between its pieces at ({}, {}, {}) (elements {ra} and {rb}): the second-order loads need a displacement continuous on the wetted surface",
                        body_label(b),
                        b.generalized[g].name,
                        fmt(x[0]),
                        fmt(x[1]),
                        fmt(x[2])
                    ))
                    .at(at_body(bi as i32)));
                }
            }
        }
        // control_surface_waterline_term: one of its two values by its type (WaterlineTerm); = 'body'
        // needs second_order.drift_control_surface: a rule of the objects
        // (document_build::check_body_objects), which build_model applies before it builds the
        // model's problem (the only one these rules are checked on)
        if opt.control_surface_drift && !opt.diffraction {
            return Err(Error::value("second_order.drift_control_surface requires the diffraction solution"));
        }
        for (bi, b) in bodies.iter().enumerate() {
            let at_b = at_body(bi as i32);
            let here = at_b;
            if !opt.control_surface_drift && b.control_surface.is_some() {
                return Err(Error::value(format!(
                    "{}: a control_surface needs second_order.drift_control_surface",
                    body_label(b)
                ))
                .at(at_b)
                .located(here));
            }
            if !opt.control_surface_drift {
                continue;
            }
            if b.control_surface.is_none() {
                return Err(Error::value(format!(
                    "{}: second_order.drift_control_surface needs a control_surface for every body",
                    body_label(b)
                ))
                .at(at_b)
                .located(here));
            }
            if !b.generalized.is_empty()
                || !b.dampers.is_empty()
                || !b.tanks.is_empty()
                || !self.fixed_dampers.is_empty()
            {
                return Err(Error::value(format!(
                    "{}: the control-surface drift is not available with generalized modes, dipole dampers or tanks",
                    body_label(b)
                ))
                .at(at_b)
                .located(here));
            }
            // the free surface inside the control surface carries the applied pressure of a pressure
            // surface there, which the control-surface formula takes as free
            if !b.pressure.is_empty() {
                return Err(Error::value(format!(
                    "{}: the control-surface drift is not available with pressure surfaces (use the momentum drift)",
                    body_label(b)
                ))
                .at(at_b)
                .located(here));
            }
        }
        if opt.control_surface_drift {
            self.control_surfaces_hold_their_body_alone()?;
        }
        // the drift of moving bodies follows their motions: without rao they would be taken as fixed; a
        // body moves in its free rigid modes, its generalized modes and its pressure modes
        if (opt.drift_angles != 0 || opt.pressure_drift || opt.control_surface_drift) && !opt.rao {
            for (bi, b) in bodies.iter().enumerate() {
                if !b.modes.is_empty() || !b.generalized.is_empty() || b.pressure.iter().any(|q| !q.modes.is_empty()) {
                    return Err(Error::value(format!(
                            "{}: the mean drift of a moving body needs its motions: set first_order.motion = true (or no free, generalized or pressure modes for a fixed body)",
                            body_label(b)
                        )).at(at_body(bi as i32)));
                }
            }
        }
        for (bi, b) in bodies.iter().enumerate() {
            if b.excitation == Excitation::Haskind && !opt.radiation {
                return Err(Error::value(format!(
                    "{}: excitation_method='haskind' needs the radiation solutions (first_order.radiation)",
                    body_label(b)
                ))
                .at(at_body(bi as i32)));
            }
        }
        if !(opt.log_tolerance.is_finite() && opt.log_tolerance >= 0.0) {
            return Err(Error::value("options.log_tolerance must be a finite number >= 0 (0: every pair)"));
        }
        // specified body points: with a body output, for every body or none
        let mut with_points = 0usize;
        for (bi, b) in bodies.iter().enumerate() {
            let here = at_body(bi as i32);
            if b.body_points.is_empty() {
                continue;
            }
            with_points += 1;
            for x in &b.body_points {
                if !finite3(x) {
                    return Err(Error::value(format!("{}: body_points must be finite", body_label(b)))
                        .at(at_body(bi as i32))
                        .located(here));
                }
            }
        }
        if with_points > 0 && !opt.body_pressure && !opt.body_velocity {
            return Err(Error::value(
                "body_points are given but no body output is requested: set first_order.body_pressure and/or first_order.body_velocity",
            ));
        }
        if with_points > 0 && with_points != bodies.len() {
            return Err(Error::value(
                "body_points must be given for every body or for none (the outputs are either at the specified points or at the default points)",
            ));
        }
        for (bi, b) in bodies.iter().enumerate() {
            if b.body_point_neighbors < 1 {
                return Err(Error::value(format!("{}: body_point_neighbors must be at least 1", body_label(b)))
                    .at(at_body(bi as i32)));
            }
        }
        if opt.frequency_threads < 1 || opt.frequency_threads > 1024 {
            return Err(Error::value(format!(
                "options.frequency_threads must be in 1..1024, got {}",
                opt.frequency_threads
            )));
        }
        if opt.inner_threads < 1 || opt.inner_threads > 1024 {
            return Err(Error::value(format!("options.inner_threads must be in 1..1024, got {}", opt.inner_threads)));
        }
        if opt.field_pressure && !self.has_field_points {
            return Err(Error::value("first_order.field_pressure requires field points"));
        }
        if opt.field_velocity && !self.has_field_points {
            return Err(Error::value("first_order.field_velocity requires field points"));
        }
        if self.has_field_points && !opt.field_pressure && !opt.field_velocity {
            return Err(Error::value(
                "field points are given but no field output is requested: set first_order.field_pressure and/or first_order.field_velocity",
            ));
        }
        for (bi, b) in bodies.iter().enumerate() {
            let here = at_body(bi as i32);
            let at = body_label(b);
            // the Haskind relation needs a jump that moves with the body, and a free damper on any body
            // removes it for every body
            if b.excitation == Excitation::Haskind
                && (!self.fixed_dampers.is_empty() || bodies.iter().any(|o| o.has_free_damper()))
            {
                return Err(
                    Error::value(format!("{at}: excitation_method='haskind' is not valid with fixed dampers")).at(here)
                );
            }
            // the names, symmetry, pieces and terms of a generalized mode are checked by the document
            // (a named list, a choice, a ModePiece needed, the terms' powers and coefficients where
            // they are set): a failure here is a bug of the problem's build
            let mut mode_names: BTreeSet<String> = BTreeSet::new();
            for g in &b.generalized {
                let w = format!("{at}: generalized mode '{}'", g.name);
                if blank(&g.name) {
                    return Err(Error::internal(format!("{at}: generalized mode names must not be blank")).at(here));
                }
                if !mode_names.insert(g.name.clone()) {
                    return Err(Error::internal(format!("{at}: duplicate generalized mode name '{}'", g.name)).at(here));
                }
                if g.parity < 1 || g.parity > 6 {
                    return Err(Error::internal(format!("{w}: parity must be a rigid mode 1..6")).at(here));
                }
                if g.pieces.is_empty() {
                    return Err(Error::internal(format!("{w} has no pieces")).at(here));
                }
                for p in &g.pieces {
                    for t in &p.terms {
                        if t.component < 0 || t.component > 2 {
                            return Err(Error::internal(format!("{w}: a term component must be u, v or w")).at(here));
                        }
                        if t.px < 0 || t.py < 0 || t.pz < 0 || t.px > 8 || t.py > 8 || t.pz > 8 {
                            return Err(Error::internal(format!("{w}: the powers of a term must be in 0..8")).at(here));
                        }
                        if !t.c.is_finite() {
                            return Err(Error::internal(format!("{w}: the coefficients must be finite")).at(here));
                        }
                    }
                }
            }
            for p in &b.pressure {
                for pm in &p.modes {
                    if mode_names.contains(&pm.name) {
                        return Err(Error::value(format!("{at}: the mode name '{}' is used twice", pm.name)).at(here));
                    }
                }
            }
            let n6 = 6 + b.generalized.len();
            let mats: [(&Vec<f64>, &str); 3] =
                [(&b.mode_mass, "mode_mass"), (&b.mode_damping, "mode_damping"), (&b.mode_stiffness, "mode_stiffness")];
            for (mat, name) in mats {
                if mat.is_empty() {
                    continue;
                }
                if b.generalized.is_empty() {
                    return Err(Error::value(format!("{at}: {name} needs generalized modes")).at(here));
                }
                if mat.len() != n6 * n6 {
                    // the matrix is stored flat and square: its side the square root of its length
                    let side = mat.len().isqrt();
                    let got = if side * side == mat.len() {
                        format!("{side} x {side}")
                    } else {
                        format!("{} values", mat.len())
                    };
                    return Err(Error::value(format!(
                        "{at}: {name} must be {n6} x {n6} (6 + G, G = {} generalized modes), not {got}",
                        n6 - 6
                    ))
                    .at(here));
                }
                for i in 0..n6 * n6 {
                    if !mat[i].is_finite() {
                        return Err(Error::value(format!("{at}: {name} must be finite")).at(here));
                    }
                    if i / n6 < 6 && i % n6 < 6 && mat[i] != 0.0 {
                        return Err(Error::value(format!(
                            "{at}: the rigid block of {name} must be zero (use the mass properties and additional_stiffness / _damping)"
                        )).at(here));
                    }
                }
            }
            for (k, d) in b.dampers.iter().enumerate() {
                let w = format!(
                    "{at}: {}",
                    if d.label.is_empty() { element_text("damper", k, &d.name) } else { d.label.clone() }
                );
                if d.panels.is_empty() {
                    return Err(Error::value(format!("{w} has no panels")).at(here));
                }
                if !(d.epsilon.is_finite() && d.epsilon >= 0.0) {
                    return Err(Error::value(format!("{w}: damping_factor must be finite and >= 0")).at(here));
                }
            }
        }
        Ok(())
    }

    /// The momentum balance of a control surface gives the force on everything inside it: every
    /// body's control surface holds that body alone (no vertex of another body inside or on it) and
    /// no damping lid (the free surface inside it is taken as free). The footprint: the convex hull
    /// of the surface's vertices in the plane.
    fn control_surfaces_hold_their_body_alone(&self) -> Result<()> {
        for (bi, b) in self.bodies.iter().enumerate() {
            let Some(cs) = b.control_surface else {
                continue;
            };
            let hull = convex_hull(plane_vertices(self, cs, Some(bi)));
            let inside = |x: [f64; 2]| hull.len() >= 3 && inside_convex(&hull, x);
            for (bj, other) in self.bodies.iter().enumerate() {
                // its mesh and the free-surface plates of meshes of their own (a piston in a tube
                // inside the hull's control surface is a second body there)
                let mut meshes = std::iter::once(other.mesh as usize).chain(other.plates.iter().filter_map(|p| p.mesh));
                if bj != bi && meshes.any(|k| plane_vertices(self, k, Some(bj)).into_iter().any(inside)) {
                    return Err(Error::value(format!(
                        "{}: its control surface reaches {} (a control surface must hold its body alone: the momentum balance gives the force on everything inside)",
                        body_label(b),
                        body_label(other)
                    ))
                    .at(at_body(bi as i32)));
                }
            }
            for l in &self.lids {
                let Some(lm) = l.mesh else {
                    continue;
                };
                let owner = (l.body >= 0).then_some(l.body as usize);
                let pts = plane_vertices(self, lm, owner);
                let n = pts.len().max(1) as f64;
                let c = [pts.iter().map(|x| x[0]).sum::<f64>() / n, pts.iter().map(|x| x[1]).sum::<f64>() / n];
                if inside(c) {
                    return Err(Error::value(format!(
                        "{}: the damping lid {} lies inside its control surface (the control-surface drift takes the free surface inside it as free)",
                        body_label(b),
                        l.name
                    ))
                    .at(at_body(bi as i32)));
                }
            }
        }
        Ok(())
    }

    // Rule 4: something to solve; radiation and diffraction both off is the
    // hydrostatics-only problem (no frequencies needed).
    pub(super) fn rule_something_to_solve(&mut self) -> Result<()> {
        if self.hydrostatics_only() {
            return Ok(());
        }
        if !self.opt.diffraction && !(self.opt.radiation && self.radiation_columns() > 0) {
            return Err(Error::value(
                "nothing to solve: enable diffraction, or radiation with at least one mode (radiation = false and diffraction = false: hydrostatics only)",
            ));
        }
        Ok(())
    }

    // Rule 5: rao requirements.
    pub(super) fn rule_rao(&mut self) -> Result<()> {
        if !self.opt.rao {
            return Ok(());
        }
        if !self.opt.radiation || !self.opt.diffraction {
            return Err(Error::value("first_order.motion = true requires radiation = true and diffraction = true"));
        }
        if self.ordinary_frequencies() == 0 {
            return Err(Error::value("first_order.motion = true requires at least one ordinary frequency"));
        }
        for (bi, b) in self.bodies.iter().enumerate() {
            let at_b = at_body(bi as i32);
            let here = at_b;
            let moving = !b.modes.is_empty();
            if moving && !b.has_cg {
                return Err(Error::value(format!(
                    "{}: first_order.motion = true requires center_of_gravity of every moving body",
                    body_label(b)
                ))
                .at(at_b)
                .located(here));
            }
            if moving && b.mass.is_none() {
                return Err(Error::value(format!(
                    "{}: first_order.motion = true requires the mass of every moving body (a number or 'displaced')",
                    body_label(b)
                ))
                .at(at_b)
                .located(here));
            }
        }
        Ok(())
    }

    // Rule 8: interior surfaces, lids, pressure surfaces in z = 0 and free-surface plates need the
    // logarithmic integration; lids need the total formulation in the higher order, thin sheets
    // the scattered one in the low order. Free-surface plates take either formulation in both
    // methods: their scattered boundary value is that of a wetted element, -dphi_I/dn (theory §9.7).
    pub(super) fn rule_integration(&mut self) -> Result<()> {
        let (mut interior, mut z0_pressure, mut plates) = (false, false, false);
        for b in self.bodies.iter() {
            if b.interior_method != InteriorMethod::None {
                interior = true;
            }
            plates |= !b.plates.is_empty();
            for p in &b.pressure {
                let m = &self.meshes[p.mesh.unwrap_or(b.mesh as usize)];
                let rec = if p.mesh.is_some() { 0 } else { p.panels[0].first - 1 };
                let s = &m.shapes[rec as usize];
                if (s.hull[0][2] + b.position[2]).abs() <= plane_tol(m, s) {
                    z0_pressure = true;
                }
            }
        }
        if !self.lids.is_empty() && self.opt.formulation != Formulation::Total && self.opt.higher_order {
            return Err(Error::value(
                "physical damping lids require first_order.diffraction_formulation='total' in the higher-order method",
            ));
        }
        // the total potential on a thin sheet rests on the discrete hypersingular
        // identity, which is poorly satisfied where the sheet meets the body
        let mut sheets = !self.fixed_dampers.is_empty();
        for d in &self.derived_ {
            sheets = sheets || d.sheet > 0;
        }
        if sheets && self.opt.diffraction && self.opt.formulation == Formulation::Total && !self.opt.higher_order {
            return Err(Error::value(
                "thin sheets (dipoles and dipole dampers) require first_order.diffraction_formulation='scattered' in the low-order method",
            ));
        }
        if !self.opt.logarithmic {
            if interior {
                return Err(Error::value("interior surfaces require options.logarithmic_integration = true"));
            }
            if !self.lids.is_empty() {
                return Err(Error::value("physical damping lids require options.logarithmic_integration = true"));
            }
            if z0_pressure {
                return Err(Error::value("pressure surfaces at z = 0 require options.logarithmic_integration = true"));
            }
            if plates {
                return Err(Error::value("free-surface plates require options.logarithmic_integration = true"));
            }
            if self.opt.domain == Domain::Channel {
                return Err(Error::value("domain 'channel' requires options.logarithmic_integration = true"));
            }
        }
        Ok(())
    }
}
