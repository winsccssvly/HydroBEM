//! The queries of a model (model.rs) on the problem built from it: the rule check and its
//! diagnostics, the route, the counts, panels, tessellation, mesh quality and motion equation of
//! the model; the queries of a body (a disabled one computed as if it were enabled) and of the
//! water; the mode displacements, the memory estimate, the status of points and empty results of
//! the present inputs.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::codes::{PanelRole, PointStatus, QuantityId, Warning};
use crate::document::ObjectId;
use crate::document_build::build_scoped;
use crate::document_enabled::{Scope as Problems, Walk, is_enabled, scoped_document, suppressed, walked_document};
use crate::error::{Diagnostic, Error, Result};
use crate::geometry::{Placement, Vec3, v3};
use crate::hydrostatics::Hydrostatics;
use crate::inspect::{Panels, Scope, Tessellation, inspect_panels, inspect_tessellation};
use crate::memory::memory_estimate;
use crate::model::{Model, ScopedProblem};
use crate::model_solve::{add_coordinates, add_field_points};
use crate::model_types::{BodyCounts, Counts, FreeSurfacePoints, MotionEquation, PointClass};
use crate::motion::motion_matrices;
use crate::problem::Problem;
use crate::quality::{MeshQuality, mesh_quality};
use crate::response::mode_displacements;
use crate::results::{Results, put};
use crate::route::Route;
use crate::solver::MemoryEstimate;
use crate::text::py_str_repr;

impl Model {
    /// The rules of the model as it is solved (its enabled objects): those between the items of an
    /// object and between objects, then the model rules (reference/inputs-rules §1); the warnings are its
    /// diagnostics. Then the refusals of the solver routes, as `route` and the solve apply them
    /// (ErrorKind::Unsupported for a valid problem without a route, the same message), so that a
    /// model that passes the check has a route; with the potential QTF, the zones of its free
    /// surface as the solve makes them (their refusals, reference/inputs-orders §3). Then each
    /// disabled object (not within another one) is checked as if it alone were enabled: what it
    /// would break is a warning (`Warning::DisabledObject`) at the place of the failure, so that a
    /// suppressed object that does not fit the model is not overlooked.
    pub fn check(&mut self) -> Result<()> {
        // the checks before a solve (`pre_solve`), as a solve applies them
        self.ensure_built()?;
        if let Err(e) = self.pre_solve() {
            return Err(self.renumbered(e));
        }
        let mut w = Vec::new();
        for id in suppressed(&self.doc, Walk::Solve) {
            let path = self.doc.obj(id).path();
            let failure = (|| -> Result<()> {
                let e = walked_document(&self.doc, Walk::Force(id))?;
                let mut p = Problem::default();
                build_scoped(&e.doc, Problems::Model, &mut p).map_err(|x| e.numbering.error(x))?;
                p.check_rules().map_err(|x| e.numbering.error(x))
            })();
            if let Err(e) = failure {
                w.push(Diagnostic::warning_at(
                    Warning::DisabledObject,
                    *e.location(),
                    format!("{path} is disabled; enabled, it would fail: {}", e.message()),
                ));
            }
        }
        self.disabled_warnings = w;
        Ok(())
    }
    /// The warnings of the last rule check, located in the model; after `check` also those of the
    /// disabled objects.
    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        let n = self.numbering();
        let mut out: Vec<Diagnostic> = self.problem.warnings.iter().map(|d| n.diagnostic(d)).collect();
        out.extend(self.disabled_warnings.iter().cloned());
        out
    }
    /// The solver route, after the model rules in their fixed order (ErrorKind::Unsupported for a
    /// valid problem without a route; `check` applies the same refusals); its name (`Route::name`)
    /// e.g. "low_order_infinite_open_sea" or "hydrostatics".
    pub fn route(&mut self) -> Result<Route> {
        self.ensure_built()?;
        match self.problem.route() {
            Ok(r) => Ok(r),
            Err(e) => Err(self.renumbered(e)),
        }
    }
    /// The sizes of the problem after the model rules.
    pub fn counts(&mut self) -> Result<Counts> {
        let p = self.checked()?;
        Ok(Counts {
            bodies: p.bodies.len(),
            meshes: p.meshes.len(),
            lids: p.lids.len(),
            ordinary_frequencies: p.ordinary_frequencies().max(0) as usize,
            headings: p.headings.len(),
            coordinates: p.coordinates().max(0) as usize,
            radiation_columns: p.radiation_columns().max(0) as usize,
            field_points: if p.has_field_points {
                (0..p.field_points.len()).filter(|&i| p.exterior_point(i)).count()
            } else {
                0
            },
            tank_points: if p.has_field_points {
                (0..p.field_points.len()).filter(|&i| !p.exterior_point(i)).count()
            } else {
                0
            },
            lid_elements: p.lid_panel_counts().iter().map(|&c| c.max(0) as usize).sum(),
            pressure_elements: p.pressure_panels().max(0) as usize,
            has_zero_frequency: p.has_zero(),
            has_infinite_frequency: p.has_infinite(),
        })
    }
    /// The elements of a body by their roles (BodyCounts), after the model rules.
    pub fn body_counts(&mut self, body: ObjectId) -> Result<BodyCounts> {
        let (index, prefix, enabled) = {
            let b = self.own_body(body)?;
            (b.index(), b.prefix(), is_enabled(b))
        };
        if !enabled {
            return Err(Error::value(format!("{prefix}the body is disabled: it is not in the model solved")));
        }
        self.checked()?;
        let body = self.numbering().bodies.iter().position(|&k| k == index).expect("an enabled body is solved");
        let p = &mut self.problem;
        let d = &p.derived()[body];
        let u = |x: i32| x.max(0) as usize;
        Ok(BodyCounts {
            stored: u(p.stored_records(p.bodies[body].mesh)),
            wetted: u(d.wet),
            wetted_with_reflections: u(d.wet * d.symmetry),
            dipole: u(d.sheet),
            interior_surface: usize::try_from(d.interior).ok(),
            pressure_modes: u(d.pressure_modes),
            generalized_modes: u(d.generalized),
            symmetry_copies: u(d.symmetry),
        })
    }
    /// The elements of every body solved (the enabled ones, in their order in the results), as
    /// `body_counts`.
    pub fn solved_body_counts(&mut self) -> Result<Vec<BodyCounts>> {
        let bodies = self.children(self.root(), "bodies")?.to_vec();
        let enabled: Vec<ObjectId> =
            bodies.into_iter().filter(|&b| self.doc.get_obj(b).is_some_and(is_enabled)).collect();
        enabled.into_iter().map(|b| self.body_counts(b)).collect()
    }
    /// The number of elements of every lid, with their reflections.
    pub fn lid_elements(&mut self) -> Result<Vec<i32>> {
        Ok(self.checked()?.lid_panel_counts())
    }
    /// The one-based coordinate ids of the radiation columns (the K axis).
    pub fn radiation_mode_ids(&mut self) -> Result<Vec<i32>> {
        Ok(self.checked()?.radiation_mode_ids())
    }
    /// The motion equation of the bodies (theory §3.1).
    pub fn motion_equation(&mut self) -> Result<MotionEquation> {
        let p = self.checked()?;
        let mm = motion_matrices(p);
        Ok(MotionEquation { ndof: 6 * p.bodies.len(), inertia: mm.M, stiffness: mm.C, damping: mm.B, ready: mm.ready })
    }
    /// The elements as the engine uses them (after a trim): global geometry and the keys (body,
    /// role, tank, owner, element, part, reflection) of every body in order and then of the
    /// elements of no body (body -1: the lids not placed with a body, the fixed dampers, the free
    /// surface); roles (empty: every role but the control surface and the free surface, the
    /// auxiliary surfaces) keeps the elements of the roles named (`PanelRole::name`: "wetted",
    /// "interior_surface", "dipole", "dipole_damper", "pressure_surface", "lid", "tank",
    /// "fixed_damper", "control_surface", "free_surface" (the panelled zone of the potential QTF's
    /// free surface as the solve integrates it: the given mesh's panels with their reflections, or
    /// the generated panels; the cells have none, see `free_surface_points`; the frequencies decide
    /// the auto sizes), "free_surface_plate"); a body's surfaces of other meshes and generated ones
    /// follow the elements of its mesh (inspect.rs `walk`); include_reflections: also the copies
    /// about the symmetry planes of the meshes. With the role "free_surface" alone, after the rules
    /// of the free surface (as `free_surface_points` of the panelled zone), otherwise every rule.
    pub fn panels(&mut self, roles: &[&str], include_reflections: bool) -> Result<Panels> {
        let roles = panel_roles(roles, false)?;
        if free_surface_only(&roles) {
            return self.free_surface_query("panelled", |p| inspect_panels(p, Scope::All, &roles, include_reflections));
        }
        inspect_panels(self.checked()?, Scope::All, &roles, include_reflections)
    }
    /// A quadrilateral mesh for display of the elements of `panels` (the same roles and
    /// include_reflections), each quad with the keys of its element: a low-order panel one quad, a
    /// patch divisions[0] x divisions[1] quads in u and v on its exact surface, 1..256 each, or
    /// [0, 0]: one quad per span of its potential discretization.
    pub fn tessellation(
        &mut self,
        roles: &[&str],
        divisions: [i32; 2],
        include_reflections: bool,
    ) -> Result<Tessellation> {
        let roles = panel_roles(roles, false)?;
        if free_surface_only(&roles) {
            return self.free_surface_query("panelled", |p| {
                inspect_tessellation(p, Scope::All, &roles, divisions, include_reflections)
            });
        }
        inspect_tessellation(self.checked()?, Scope::All, &roles, divisions, include_reflections)
    }
    /// The quality of the body meshes: per element the panel or span size, aspect ratio, panels or
    /// spans per shortest wavelength and the quality flags; per body the volumes and the edge
    /// counts of the closed surface.
    pub fn mesh_quality(&mut self) -> Result<MeshQuality> {
        mesh_quality(self.checked()?, -1)
    }

    // ---- the queries of a body (after the rules of that body alone: the environment, its geometry
    // and its surfaces; those of the options, the orders, the frequencies, the field points and
    // the other bodies do not concern them, reference/queries)

    // the problem of a scope's queries (`Scope::Body`, `Scope::FreeSurface`) after the rules of its
    // objects and of its problem (`build_scoped`, `Problem::check_scope`), errors located in the
    // model. It is kept (`Model::scoped`) for the next query of the same scope until an input
    // changes: every change of the document moves its revision on.
    pub(crate) fn scoped_query(&mut self, scope: Problems) -> Result<&mut ScopedProblem> {
        if self.calculating.load(Ordering::SeqCst) {
            return Err(Error::state("the model is calculating: it can be queried when the solve ends"));
        }
        let revision = self.doc.revision();
        let kept = self.scoped.as_ref().is_some_and(|k| k.revision == revision && k.scope == scope);
        if !kept {
            self.scoped = None;
            let rules = scope.rules().ok_or_else(|| Error::internal("a scope of queries builds a problem"))?;
            let e = scoped_document(&self.doc, scope)?;
            let mut p = Problem::default();
            build_scoped(&e.doc, scope, &mut p).map_err(|x| e.numbering.error(x))?;
            p.check_scope(rules).map_err(|x| e.numbering.error(x))?;
            let index = match scope {
                Problems::Body(body) => {
                    let name = self.doc.obj(body).name().to_string();
                    e.doc.root().children("bodies").iter().position(|b| b.name() == name).unwrap_or(0) as i32
                }
                _ => 0,
            };
            self.scoped = Some(ScopedProblem { revision, scope, numbering: e.numbering, problem: Box::new(p), index });
        }
        Ok(self.scoped.as_mut().expect("the problem of the scope"))
    }
    // a query of a body: f on the problem of the body alone (`Scope::Body`: the body taken as
    // enabled, without the other bodies, the lids placed with them and the motions; the bodies
    // whose trimmed waterline loops its generated plates cover kept, in the model's order) and the
    // body's index there, after the rules of the body's objects and of its problem (the
    // environment, the geometry, the surfaces). Errors located in the model.
    pub(crate) fn body_query<T>(
        &mut self,
        body: ObjectId,
        f: impl FnOnce(&mut Problem, i32) -> Result<T>,
    ) -> Result<T> {
        self.own_body(body)?;
        let k = self.scoped_query(Problems::Body(body))?;
        f(&mut k.problem, k.index).map_err(|x| k.numbering.error(x))
    }
    // the numbering of the rows of a body's queries in the model as it is solved, the body taken
    // as enabled (as the rows of the model's queries): the body's index among the enabled bodies,
    // and per lid of the body's problem its index among the enabled lids
    fn body_rows(&self, body: ObjectId) -> (i32, Vec<i32>) {
        let root = self.doc.root();
        let bodies = root.children("bodies");
        let index = bodies.iter().take_while(|b| b.id() != body).filter(|b| is_enabled(**b)).count() as i32;
        let lids = root.children("lids");
        let enabled_before = |j: usize| lids[..j].iter().filter(|l| is_enabled(**l)).count() as i32;
        let kept = self.scoped.as_ref().map(|k| k.numbering.lids.clone()).unwrap_or_default();
        (index, kept.into_iter().map(enabled_before).collect())
    }
    // the rows of a body's query numbered in the model (`body_rows`): the body, and the owner of
    // a lid's elements
    fn renumber_rows(&self, body: ObjectId, bodies: &mut [i32], rows: Option<(&[PanelRole], &mut [i32])>) {
        let (index, lids) = self.body_rows(body);
        let own = self.scoped.as_ref().map_or(0, |k| k.index);
        for b in bodies.iter_mut().filter(|b| **b == own) {
            *b = index;
        }
        if let Some((roles, owners)) = rows {
            for (r, o) in roles.iter().zip(owners.iter_mut()) {
                if *r == PanelRole::Lid && *o >= 0 {
                    *o = lids.get(*o as usize).copied().unwrap_or(*o);
                }
            }
        }
    }
    /// The hydrostatics of a body (theory §3.2) after its trim and placement, body axes about R: the
    /// volumes, the displaced mass, the centre of buoyancy, the waterplane, the stiffness without and
    /// with the weight terms, the metacentric heights with the weight terms. A disabled body is
    /// computed as if it were enabled (also the other queries of a body).
    pub fn body_hydrostatics(&mut self, body: ObjectId) -> Result<Hydrostatics> {
        self.body_query(body, |p, bi| Ok(p.hydrostatics(bi)))
    }
    /// The elements of a body as the engine uses them (as `panels`, of this body alone; no
    /// fixed_damper or free_surface, which are elements of no body).
    pub fn body_panels(&mut self, body: ObjectId, roles: &[&str], include_reflections: bool) -> Result<Panels> {
        let roles = panel_roles(roles, true)?;
        let mut x = self.body_query(body, |p, bi| inspect_panels(p, Scope::Body(bi), &roles, include_reflections))?;
        self.renumber_rows(body, &mut x.body, Some((&x.role, &mut x.owner)));
        Ok(x)
    }
    /// The quality of the mesh of a body (as `mesh_quality`, of this body alone).
    pub fn body_mesh_quality(&mut self, body: ObjectId) -> Result<MeshQuality> {
        let mut x = self.body_query(body, |p, bi| mesh_quality(p, bi))?;
        self.renumber_rows(body, &mut x.body, None);
        Ok(x)
    }
    /// The tessellation of the elements of a body as the engine uses them, placed (as
    /// `tessellation`, of this body alone).
    pub fn body_tessellation(
        &mut self,
        body: ObjectId,
        roles: &[&str],
        divisions: [i32; 2],
        include_reflections: bool,
    ) -> Result<Tessellation> {
        let roles = panel_roles(roles, true)?;
        let mut x = self.body_query(body, |p, bi| {
            inspect_tessellation(p, Scope::Body(bi), &roles, divisions, include_reflections)
        })?;
        let roles: Vec<PanelRole> = x.role.iter().map(|r| r.unwrap_or(PanelRole::Wetted)).collect();
        self.renumber_rows(body, &mut x.body, Some((&roles, &mut x.owner)));
        Ok(x)
    }
    /// The points and weights (m^2, or m along the outer circle) of one zone of the potential QTF's
    /// free surface as the solve integrates it (second_order.free_surface_*; theory §12.8-§12.9), global:
    /// "panelled" (the panelled zone: the panels' Gauss points or the cells'), "quadrature" (the
    /// quadrature zone) or "outer_circle" (the line integral around the outer circle). The
    /// frequencies decide the auto sizes and the order. After the rules of the free surface alone
    /// (`free_surface_query`): a zone whose auto values the frequencies decide is refused without a
    /// frequency between 0 and inf.
    pub fn free_surface_points(&mut self, zone: &str) -> Result<FreeSurfacePoints> {
        if !matches!(zone, "panelled" | "quadrature" | "outer_circle") {
            return Err(Error::value(format!(
                "zone must be \"panelled\", \"quadrature\" or \"outer_circle\", got \"{zone}\""
            )));
        }
        let z = self.free_surface_query(zone, |p| crate::potential_qtf::zones(p))?;
        let pick = |r: std::ops::Range<usize>| {
            let s = &z.pts[r];
            FreeSurfacePoints {
                points: s.iter().map(|(x, _)| [x[0], x[1], x[2]]).collect(),
                weights: s.iter().map(|(_, w)| *w).collect(),
            }
        };
        match zone {
            "panelled" => Ok(pick(0..z.n_inner)),
            "quadrature" => Ok(pick(z.n_inner..z.pts.len())),
            "outer_circle" => {
                let r = z.outer_radius;
                let dl = 2.0 * crate::constants::PI * r / z.line_points as f64;
                let pts = (0..z.line_points)
                    .map(|t| {
                        let th = z.line_angle(t);
                        [z.c0[0] + r * th.cos(), z.c0[1] + r * th.sin(), 0.0]
                    })
                    .collect();
                Ok(FreeSurfacePoints { points: pts, weights: vec![dl; z.line_points] })
            }
            _ => unreachable!("the zone was checked above"),
        }
    }
    // a free-surface query: f on the problem of the free surface (`Scope::FreeSurface`: the enabled
    // objects, every body, without the motions) after the rules of the free surface
    // (`RuleScope::FreeSurface`: the free-surface items, the geometry, the surfaces, the
    // environment and its domain); where the auto values of `zone` take the frequencies
    // (`frequency_items`), also the values of the frequencies and at least one between 0 and inf.
    // Errors located in the model.
    pub(crate) fn free_surface_query<T>(&mut self, zone: &str, f: impl FnOnce(&mut Problem) -> Result<T>) -> Result<T> {
        let input = self.doc.root().child("environment", "").map(|f| f.get("frequency_input").text().to_string());
        let k = self.scoped_query(Problems::FreeSurface)?;
        let items = frequency_items(&k.problem, zone);
        if !items.is_empty() {
            k.problem.rule_frequency_values().map_err(|x| k.numbering.error(x))?;
            if k.problem.ordinary_frequencies() == 0 {
                return Err(Error::value(format!(
                    "environment: {} has no frequency between 0 and inf, and the auto {} of the {zone} zone of the potential QTF's free surface {} from the frequencies (give the frequencies, or the value in second_order)",
                    input.unwrap_or_else(|| "omegas".to_string()),
                    items.iter().map(|i| format!("second_order.{i}")).collect::<Vec<_>>().join(" and "),
                    if items.len() == 1 { "is" } else { "are" }
                )));
            }
        }
        f(&mut k.problem).map_err(|x| k.numbering.error(x))
    }
    /// Displacements of every coordinate at points of the bodies: per point (global), its
    /// zero-based body and the one-based stored element of its mesh (the numbering of the mesh
    /// references and of `Panels::element`) and the point (global); n x D x 3, global axes (D =
    /// Counts::coordinates). An element the body does not use (trimmed away, or of no role) is
    /// refused.
    pub fn mode_displacements(&mut self, points: &[[f64; 3]], bodies: &[i32], elements: &[i32]) -> Result<Vec<f64>> {
        if bodies.len() != points.len() || elements.len() != points.len() {
            return Err(Error::argument("bodies, elements and points must have one entry per point"));
        }
        let flat: Vec<f64> = points.iter().flatten().copied().collect();
        let n = bodies.len();
        self.checked()?;
        let numbering = self.numbering(); // of the model as checked now
        let p = &mut self.problem;
        let mut out = vec![0.0; n * p.coordinates() as usize * 3];
        mode_displacements(p, n as i32, bodies, elements, &flat, &mut out).map_err(|e| numbering.error(e))?;
        Ok(out)
    }
    /// An estimate of the memory of a solve without solving: the model is validated and its geometry
    /// built; the counts of threads are those of the options. The tank problems and the
    /// copies of the results are not included.
    pub fn memory_estimate(&mut self) -> Result<MemoryEstimate> {
        self.ensure_built()?;
        let n = self.numbering();
        memory_estimate(&mut self.problem).map_err(|e| n.error(e))
    }
    /// The status of points by the field-point rules of the model (reference/inputs-model-objects §3), whether or not
    /// they are its field points: its geometry and surfaces must be valid (its frequencies, options and
    /// own field points are not checked). tank: None, points in the sea (global coordinates, as
    /// first_order.field_points); or a tank of the model, points inside it (body coordinates, as its
    /// field_points; a tank the solve does not take, disabled or of a disabled body, is the status
    /// NoTank). A point gets the zero-based body of its status (on, near or inside a body, inside
    /// its waterplane), the lid it is on, or for a point given to a tank (`Tank`, `EmptyTank`,
    /// `AboveTankSurface`) the tank's body and the zero-based tank of that body, as the results
    /// number a body's tanks (`bodies[b].tanks[t]`).
    pub fn classify_points(&mut self, points: &[[f64; 3]], tank: Option<ObjectId>) -> Result<Vec<PointClass>> {
        // the tank's body places its points; its number in the problem (-1: not solved)
        let mut place = Placement { position: v3(0.0, 0.0, 0.0), yaw_deg: 0.0 };
        if let Some(t) = tank {
            let o = self.obj(t)?;
            let Some(b) = o.parent().filter(|_| o.type_().name == "Tank") else {
                return Err(Error::argument("tank: not a tank of this model"));
            };
            let p = b.get("position").reals().to_vec();
            place = Placement { position: v3(p[0], p[1], p[2]), yaw_deg: b.get("yaw").real() };
        }
        self.ensure_built()?;
        let n = self.numbering();
        let mut within = None; // the tank's zero-based place among the tanks of its body
        let mut tank_body = None; // the tank's body among the bodies solved
        let tanks: Vec<i32> = match tank {
            None => Vec::new(),
            Some(t) => {
                let doc = &self.enabled.as_ref().expect("built").doc;
                let mut k = 0;
                let mut number = -1;
                for (bk, b) in doc.root().children("bodies").iter().enumerate() {
                    for (j, x) in b.children("tanks").iter().enumerate() {
                        k += 1;
                        if x.id() == t {
                            number = k;
                            within = Some(j);
                            tank_body = Some(bk);
                        }
                    }
                }
                vec![number; points.len()]
            }
        };
        let m = &mut self.problem;
        // the geometry rules alone, then the warnings of the last full check back
        let warnings = m.warnings.clone();
        m.check_geometry_rules().map_err(|e| n.error(e))?;
        m.invalidate(); // the other rules were not checked
        m.warnings = warnings;
        let pts: Vec<Vec3> =
            points.iter().map(|p| if tank.is_some() { place.apply(Vec3(*p)) } else { Vec3(*p) }).collect();
        Ok(m.classify_field_points(&pts, &tanks)
            .iter()
            .map(|s| {
                let of_tank =
                    matches!(s.status, PointStatus::Tank | PointStatus::EmptyTank | PointStatus::AboveTankSurface);
                let index = usize::try_from(s.index).ok();
                match s.status {
                    _ if of_tank => PointClass { status: s.status, body: tank_body, lid: None, tank: within },
                    PointStatus::OnLid => PointClass { status: s.status, body: None, lid: index, tank: None },
                    _ => PointClass { status: s.status, body: index, lid: None, tank: None },
                }
            })
            .collect())
    }
    /// Empty results of the present inputs, to be filled by `Results::put` (results computed
    /// elsewhere, to write them): a copy of the inputs, the axes of the model (omegas, headings,
    /// radiation_mode_ids, coordinates) and its points of the field outputs (field_points, tanks and
    /// tanks_ids, when it has them).
    pub fn new_results(&mut self) -> Result<Results> {
        let doc = Arc::new(self.doc.clone());
        let m = self.checked()?;
        let mut r = Results { route: "given".to_string(), model: Some(doc), ..Results::default() };
        let mut om = Vec::new();
        for &w in &m.omegas {
            if w > 0.0 && w.is_finite() {
                om.push(w);
            }
        }
        put(&mut r, QuantityId::Omegas, vec![om.len() as i64], &om);
        put(&mut r, QuantityId::Headings, vec![m.headings.len() as i64], &m.headings);
        let ids = m.radiation_mode_ids();
        put(&mut r, QuantityId::RadiationModeIds, vec![ids.len() as i64], &ids);
        add_coordinates(m, &mut r);
        let solved = r.solved_document().ok_or_else(|| Error::internal("new results hold the model"))?;
        add_field_points(&solved, &mut r);
        Ok(r)
    }
}

// the roles of a free-surface query: the free surface alone
fn free_surface_only(roles: &[PanelRole]) -> bool {
    !roles.is_empty() && roles.iter().all(|&r| r == PanelRole::FreeSurface)
}

// the items of second_order whose auto values the frequencies decide in a zone of the free surface
// (reference/inputs-orders §3): the size of the generated panels or cells of the panelled zone; the
// radial step and the azimuthal nodes of the quadrature zone (with annuli); the radius of the outer
// circle (the annuli's steps) and its segments
fn frequency_items(p: &Problem, zone: &str) -> Vec<&'static str> {
    let o = &p.opt;
    let annuli = o.potential_qtf_annuli > 0;
    let step = annuli && o.potential_qtf_annulus_step.is_none();
    let candidates: &[(bool, &'static str)] = match zone {
        "panelled" => {
            &[(o.free_surface_mesh.is_none() && o.potential_qtf_panel_size.is_none(), "free_surface_panel_size")]
        }
        "quadrature" => &[
            (step, "quadrature_radius_step"),
            (annuli && o.potential_qtf_azimuthal_nodes.is_none(), "quadrature_azimuthal_nodes"),
        ],
        _ => &[(step, "quadrature_radius_step"), (o.potential_qtf_segments.is_none(), "outer_circle_segments")],
    };
    candidates.iter().filter(|(on, _)| *on).map(|(_, item)| *item).collect()
}

// the roles of the panels and tessellation queries (empty: the default roles) of every body and
// the elements of no body, or of a body (body: it has neither the fixed dampers nor the free
// surface, which belong to no body)
fn panel_roles(roles: &[&str], body: bool) -> Result<Vec<PanelRole>> {
    let of_no_body = |r: PanelRole| matches!(r, PanelRole::FixedDamper | PanelRole::FreeSurface);
    let names: Vec<&str> = PanelRole::ALL.iter().filter(|&&r| !(body && of_no_body(r))).map(|r| r.name()).collect();
    let mut out = Vec::new();
    for &role in roles {
        let found = PanelRole::from_name(role);
        match found.filter(|&r| !(body && of_no_body(r))) {
            Some(r) => {
                if !out.contains(&r) {
                    out.push(r);
                }
            }
            None => {
                return Err(Error::value(format!(
                    "role must be one of {}, got {}{}",
                    names.join(", "),
                    py_str_repr(role),
                    if found.is_some() {
                        " (the fixed dampers and the free surface are elements of the model, not of a body: the query of the model)"
                    } else {
                        ""
                    }
                )));
            }
        }
    }
    Ok(out)
}
