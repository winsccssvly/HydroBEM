//! The solver routes: the route of a valid problem, the reasons the higher order cannot solve
//! one, and the wavemakers in the walls.
//!
//! Solver routes: which implemented method solves a valid model (both methods), or the reason
//! why none does (ErrorKind::Unsupported). The route name is <method>_<infinite|finite>_<domain>.

use crate::constants::PI;
use crate::error::{Error, Result, at_body};
use crate::geometry::{COINCIDENCE_TOL, Panel, Vec3, dot, length_tol};
use crate::ho_internal::MAX_MODES;
use crate::mesh::Mesh;
use crate::mesh_desc::{AnalyticKind, PatchKind, Profile};
use crate::model_codes::{Domain, InteriorMethod};
use crate::problem::{Problem, Role};

/// The method of a route (non-exhaustive: a later method is not a breaking change).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// The low-order method (flat panels, constant potential).
    LowOrder,
    /// The higher-order method (B-spline potential on patches).
    HigherOrder,
}

/// The solver route of a valid problem (`Model::route`): the hydrostatics alone (no
/// boundary-value problem), or a method with the depth and the domain it solves in. Its name
/// (`Route::name`, also its `Display`) is what the results' meta data and the `.hbr` hold, C
/// `hb_model_route` and Python `m.route()` give; a route equals its name as any string type
/// (`route == "hydrostatics"`, a `str` or a `String`, either side). Non-exhaustive, as `Method` and
/// `Domain`: a match outside the crate has a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Route {
    /// No boundary-value problem: the hydrostatics alone.
    Hydrostatics,
    /// A boundary-value problem solved by `method` in finite depth or not, in `domain`.
    Solve { method: Method, finite_depth: bool, domain: Domain },
}

impl std::fmt::Display for Route {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

impl PartialEq<&str> for Route {
    fn eq(&self, name: &&str) -> bool {
        self.name() == *name
    }
}

impl PartialEq<Route> for &str {
    fn eq(&self, route: &Route) -> bool {
        route.name() == *self
    }
}

impl PartialEq<str> for Route {
    fn eq(&self, name: &str) -> bool {
        self.name() == name
    }
}

impl PartialEq<Route> for str {
    fn eq(&self, route: &Route) -> bool {
        route.name() == self
    }
}

impl PartialEq<String> for Route {
    fn eq(&self, name: &String) -> bool {
        self.name() == *name
    }
}

impl PartialEq<Route> for String {
    fn eq(&self, route: &Route) -> bool {
        route.name() == *self
    }
}

impl Route {
    /// The route of a problem solved by `method`, in its depth and domain.
    fn of(m: &Problem, method: Method) -> Route {
        Route::Solve { method, finite_depth: m.env.depth.is_finite(), domain: m.opt.domain }
    }

    /// The name of the route: "hydrostatics", or <method>_<infinite|finite>_<domain>, e.g.
    /// "low_order_infinite_open_sea", "higher_order_finite_x_wall".
    pub fn name(&self) -> String {
        let Route::Solve { method, finite_depth, domain } = *self else { return "hydrostatics".into() };
        let method = match method {
            Method::LowOrder => "low_order",
            Method::HigherOrder => "higher_order",
        };
        let domain = match domain {
            Domain::OpenSea => "open_sea",
            Domain::XWall => "x_wall",
            Domain::YWall => "y_wall",
            Domain::OrthogonalWalls => "orthogonal_walls",
            Domain::Channel => "channel",
        };
        format!("{method}{}{domain}", if finite_depth { "_finite_" } else { "_infinite_" })
    }
}

/// The solver route of a valid problem.
/// m: a model that passed the model rules; returns its `Route` (named e.g.
/// "low_order_infinite_open_sea", "higher_order_finite_open_sea", "hydrostatics"); an
/// ErrorKind::Unsupported error with the reason when no route exists.
pub fn select_route(m: &Problem) -> Result<Route> {
    fn no(why: &str) -> Error {
        Error::unsupported(format!("no solver route for this problem in this build: {why}"))
    }
    if m.hydrostatics_only() {
        return Ok(Route::Hydrostatics);
    }
    plates_unsupported(m)?;
    crate::tanks::tank_plates_unsupported(m)?;
    if !m.opt.higher_order
        && (m.opt.pressure_drift || m.opt.control_surface_drift || m.opt.field_velocity || m.opt.body_velocity)
    {
        // the source formulation for the velocities: walls as image panels,
        // interior surfaces and lids, pressure surfaces and a channel; the
        // sources cannot represent a dipole sheet (nor the control-surface drift)
        for b in &m.bodies {
            if b.has_sheets() || !m.fixed_dampers.is_empty() {
                return Err(no(
                    "the low-order pressure-integration and control-surface drift, field velocity and body velocity \
                    with thin sheets are not available",
                ));
            }
        }
    }
    if m.opt.drift_angles != 0 {
        // the momentum drift (theory §11.1) in open sea without lids
        if m.opt.domain != Domain::OpenSea {
            return Err(no("the mean drift with walls or a channel is not available"));
        }
        if !m.lids.is_empty() {
            return Err(no("the mean drift with damping lids is not available"));
        }
    }
    {
        // wavemakers in the walls (theory §10.3): the radiation of their modes
        // alone, from the known source strengths
        let nw = (0..m.bodies.len()).filter(|&b| wall_wavemaker(m, b)).count();
        if nw > 0 {
            if nw != m.bodies.len() {
                return Err(no("wavemakers in the walls together with other bodies are not available"));
            }
            if m.opt.diffraction {
                return Err(no("wavemakers in the walls: the radiation problem only (diffraction off)"));
            }
            if m.opt.rao
                || m.opt.pressure_drift
                || m.opt.control_surface_drift
                || m.opt.drift_angles != 0
                || m.opt.field_velocity
                || m.opt.body_velocity
            {
                return Err(no("wavemakers in the walls: motions, mean drift and velocities are not available"));
            }
            if !m.lids.is_empty() {
                return Err(no("wavemakers in the walls with damping lids are not available"));
            }
            for b in &m.bodies {
                if b.has_sheets()
                    || !m.fixed_dampers.is_empty()
                    || !b.pressure.is_empty()
                    || b.interior_method != InteriorMethod::None
                    || !b.tanks.is_empty()
                {
                    return Err(no(
                        "wavemakers in the walls with sheets, pressure or interior surfaces or tanks are not available",
                    ));
                }
            }
        }
    }
    if m.opt.domain == Domain::Channel {
        // channel (both methods): finite depth (rule 7); omega = 0 with the zero-frequency image
        // sums; W >= h / pi keeps the image series in its domain
        if m.opt.channel_width < m.env.depth / PI {
            return Err(no("a channel narrower than depth / pi is not available"));
        }
    }
    if m.opt.higher_order {
        let mut body = -1;
        let why = ho_unsupported(m, Some(&mut body));
        let here = at_body(body);
        if !why.is_empty() {
            return Err(no(&why).at(here));
        }
        return Ok(Route::of(m, Method::HigherOrder));
    }
    let mut sheets = !m.fixed_dampers.is_empty();
    let mut pressure = false;
    for b in &m.bodies {
        if b.has_sheets() {
            sheets = true;
        }
        if !b.pressure.is_empty() {
            pressure = true;
        }
    }
    if pressure {
        // prescribed-pressure surfaces: named shapes (with interior surfaces, sheets and lids);
        // omega = inf only for surfaces in z = 0
        for (bi, b) in m.bodies.iter().enumerate() {
            let here = at_body(bi as i32);
            if b.pressure.is_empty() {
                continue;
            }
            for p in &b.pressure {
                if m.has_infinite() {
                    let pmesh = &m.meshes[p.mesh.unwrap_or(b.mesh as usize)];
                    let (z, d);
                    if p.mesh.is_some() {
                        z = pmesh.panels[0].centroid[2];
                        d = pmesh.panels[0].diameter;
                    } else {
                        let r = &p.panels[0];
                        z = pmesh.panels[(r.first - 1) as usize].centroid[2];
                        d = pmesh.panels[(r.first - 1) as usize].diameter;
                    }
                    if (z + b.position[2]).abs() > COINCIDENCE_TOL * d {
                        return Err(
                            no("the omega = inf limit with a submerged pressure surface is not available").at(here)
                        );
                    }
                }
            }
        }
    }
    if sheets {
        // thin sheets: any depth, open sea or walls (with interior surfaces, pressure surfaces and
        // lids; Haskind with the jump term of the sheet, theory §9.5). A sheet panel in a symmetry
        // plane of the mesh that reflects it would coincide with its own image.
        let in_plane = |mesh: &Mesh, p: &Panel| {
            let tol = COINCIDENCE_TOL * p.diameter;
            (mesh.isx != 0 && p.centroid[0].abs() <= tol && p.normal[0].abs() > 0.5)
                || (mesh.isy != 0 && p.centroid[1].abs() <= tol && p.normal[1].abs() > 0.5)
        };
        for bi in 0..m.bodies.len() {
            let here = at_body(bi as i32);
            let b = &m.bodies[bi];
            // the body mesh's symmetry planes, also for the panels of its dipole mesh
            let mesh: &Mesh = &m.meshes[b.mesh as usize];
            for r in 0..mesh.count as usize {
                if m.derived()[bi].role[r] == Role::Dipole && in_plane(mesh, &mesh.panels[r]) {
                    return Err(no("a thin sheet in a symmetry plane of its mesh is not available").at(here));
                }
            }
            if let Some(dm) = b.dipole_mesh {
                for p in &m.meshes[dm].panels {
                    if in_plane(mesh, p) {
                        return Err(no("a thin sheet in a symmetry plane of its mesh is not available").at(here));
                    }
                }
            }
        }
        for d in &m.fixed_dampers {
            let mesh: &Mesh = &m.meshes[d.mesh as usize];
            for p in &mesh.panels {
                if in_plane(mesh, p) {
                    return Err(no("a thin sheet in a symmetry plane of its mesh is not available"));
                }
            }
        }
    }
    Ok(Route::of(m, Method::LowOrder))
}

/// The refusals of the free-surface plates (theory §9.7): available with radiation, diffraction,
/// motions, field and body outputs at ordinary frequencies, omega = 0 and omega = inf, in the open
/// sea, by walls or in a channel, on bodies with internal tanks (the tank problems never see the
/// plates, theory §10.2), with the mean drift by momentum conservation and Newman's approximation
/// of it (theory §11.1, §12.13), and, for plates whose every edge lies on a wall (of their own
/// body or of one they slide along) or between plate elements, the mean drift by pressure
/// integration and by the control surface, their quadratic QTFs and Newman's approximation of them
/// (theory §11.2, §11.3, §12.3), both methods, and the potential QTF of the low order, both
/// methods, with the free surface of the generated panels or a given mesh (theory §12.7, §12.8);
/// not a plate edge bordered by free surface in these near-field drifts (their line terms need a
/// model of the edge) or in the potential QTF (its forcing is not integrable there), pressure
/// surfaces with the pressure drift, the cells of the potential QTF's free surface, a thin sheet
/// hanging from a plate at omega = inf, wavemakers in the walls or a plate lying on a tank's fluid (a
/// floating roof; the plates of a tank's free surface are the tank's, `tanks::tank_plates_unsupported`).
/// A plate body inside another body's control surface is refused by the model
/// rules, as any second body there.
fn plates_unsupported(m: &Problem) -> Result<()> {
    let Some(first) = m.bodies.iter().position(|b| !b.plates.is_empty()) else { return Ok(()) };
    // located at the body of the plate the refusal names (else at the first body with plates)
    let no_at = |bi: i32, why: &str| {
        Err(Error::unsupported(format!("no solver route for this problem in this build: free-surface plates {why}"))
            .at(at_body(bi)))
    };
    let no = |why: &str| no_at(first as i32, why);
    let o = &m.opt;
    // a plate of the sea over a tank's fluid (a floating roof as a body of its own) would couple a
    // body to another body's tank fluid; a tank's plates are its own (theory §10.2)
    if let Some((plate, tank, bi)) = crate::plate_rules::plate_on_tank(m) {
        return no_at(
            bi,
            &format!(
                "on the free surface of a tank are not available as plates of the sea or a body of their own \
             ({plate} lies on the fluid of {tank}, a floating roof): a plate of a tank's free surface is a \
             plate of the tank (tanks[...].free_surface_plates), moving with its body"
            ),
        );
    }
    // at omega = inf a plate is a sheet in z = 0 of the odd-image problem (theory §9.8): a sheet
    // hanging from it makes a junction of three sheets (the sheet, its image and the plate)
    if m.has_infinite()
        && let Some((sheet, plate, bi)) = crate::plate_rules::sheet_on_plate(m)
    {
        return Err(Error::unsupported(format!(
            "no solver route for this problem in this build: the omega = inf limit with a thin sheet hanging \
             from a free-surface plate is not available ({sheet} meets {plate} in z = 0)"
        ))
        .at(at_body(bi)));
    }
    // the second-order potential with plates whose edges lie on walls (theory §12.7, §12.8): its
    // free-surface forcing is not integrable at a plate edge bordered by free surface (theory §9.7),
    // so a free edge is refused for good; the free surface is the generated panels or a given mesh
    // (the cells of an automatic control surface do not end at the plates)
    if o.potential_qtf.any() {
        if let Some((plate, element, bi)) = crate::plate_rules::free_plate_edge(m) {
            return no_at(
                bi,
                &format!(
                    "with the potential QTF are not available for a plate with a free edge, one bordered by free \
                 surface ({plate}, element {element}): the second-order free-surface forcing is not integrable \
                 at such an edge (the potential QTF takes plates whose edges lie on walls)"
                ),
            );
        }
        if o.free_surface_mesh.is_none() && !o.potential_qtf_panels {
            return no("with the potential QTF take the free surface of the generated panels or of a given mesh \
                 (second_order.free_surface_method = 'panels' or 'mesh'), not the cells");
        }
    }
    // the near-field drifts and their QTFs (the quadratic ones switch the drifts on, Newman's
    // approximation is made from the drifts as returned) need the pressure or the free surface at
    // the plate's edges, singular at an edge bordered by free surface: only plates whose edges lie
    // on walls (theory §11.2, §11.3)
    let pressure = o.pressure_drift || o.pressure_qtf.any();
    let control = o.control_surface_drift || o.control_surface_qtf;
    if pressure || control {
        if let Some((plate, element, bi)) = crate::plate_rules::free_plate_edge(m) {
            let what = if pressure {
                "by pressure integration (and its QTFs)"
            } else {
                "by the control surface (and its QTFs)"
            };
            return no_at(
                bi,
                &format!(
                    "with the mean drift {what} are not available for a plate with a free edge, one bordered by free \
                 surface ({plate}, element {element}); the mean drift by momentum conservation is"
                ),
            );
        }
        if pressure && m.bodies.iter().any(|b| !b.pressure.is_empty()) {
            return no("with the mean drift by pressure integration (and its QTFs) and pressure surfaces are not \
                 available");
        }
    }
    if (0..m.bodies.len()).any(|b| wall_wavemaker(m, b)) {
        return no("with wavemakers in the walls are not available");
    }
    Ok(())
}

/// A body lying wholly in the wall plane(s) of the domain: a wavemaker in the wall
/// (theory §10.3), every element in x = 0 (x wall) or y = 0 (y wall).
/// b: the zero-based body.
pub fn wall_wavemaker(m: &Problem, b: usize) -> bool {
    m.derived()[b].wavemaker
}

/// Reason why the higher-order route cannot solve a valid problem.
/// m: problem with options.higher_order; body: set to the body concerned when given. Returns the reason, or "" when the route can solve it.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn ho_unsupported(m: &Problem, mut body: Option<&mut i32>) -> String {
    if let Some(b) = body.as_deref_mut() {
        *b = -1;
    }
    if m.coordinates() > MAX_MODES as i32 {
        return format!(
            "more than {} higher-order coordinates (6 per body, the generalized and the pressure modes) are not available",
            MAX_MODES
        );
    }
    for l in &m.lids {
        if l.body < 0 {
            return "damping lids as external meshes need a patch mesh of the first body in the higher-order method \
                    (they join its elements)"
                .into();
        }
    }
    if !m.fixed_dampers.is_empty() {
        return "higher-order fixed dampers are available as elements of the mesh of a body at the origin \
                 without yaw or trim (a fixed damper on a mesh of its own is not available)"
            .into();
    }
    for bi in 0..m.bodies.len() {
        if let Some(b) = body.as_deref_mut() {
            *b = bi as i32; // the body of a reason returned in this loop
        }
        let b = &m.bodies[bi];
        if !m.meshes[b.mesh as usize].patch {
            return "the higher-order method needs patch meshes for every body".into();
        }
        // an analytic map with a vanishing slope at the waterline makes the tangential
        // derivative of the potential singular there: |grad phi|^2 is not integrable
        let mb = &m.meshes[b.mesh as usize];
        if m.opt.pressure_drift && mb.geometry == PatchKind::Analytic && !mb.trimmed() {
            for ap in &mb.analytic {
                if ap.kind == AnalyticKind::Bilinear || ap.profile == Profile::Linear {
                    continue;
                }
                // either end v = -1 or v = 1 of the profile with dP = 0 that lies on the waterline of
                // the placed body (z + position z = 0)
                for v in [-1.0, 1.0] {
                    let (mut P, mut dP) = (0.0, 0.0);
                    ap.profile_of(0.5 * (v + 1.0), &mut P, &mut dP);
                    if dP != 0.0 {
                        continue;
                    }
                    let mut waterline = true;
                    for u in [-1.0, 0.0, 1.0] {
                        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                        ap.eval(u, v, &mut x, &mut xu, &mut xv);
                        waterline = waterline && (x[2] + b.position[2]).abs() <= length_tol(mb.reference_length);
                    }
                    if waterline {
                        return "the pressure-integration drift with a nonuniform analytic mapping at the waterline is \
                                not available (use the uniform mapping or the momentum drift)"
                            .into();
                    }
                }
            }
        }
        // interior free surfaces and pressure surfaces as patches of the body mesh
        if b.interior_method != InteriorMethod::None && b.interior_method != InteriorMethod::PanelRange {
            return "higher-order interior surfaces other than patches of the body mesh or of a patch mesh of \
                    their own (or the waterline surface of a trimmed body) are not available"
                .into();
        }
        if b.dipole_mesh.is_some() {
            return "higher-order thin sheets need patches of the body mesh or of a patch mesh of their own".into();
        }
        if !b.dipole_panels.is_empty() {
            // with interior surfaces, pressure surfaces and lids
            if m.opt.pressure_drift {
                return "the pressure-integration drift with thin sheets is not available (no velocity on a sheet)"
                    .into();
            }
            let ms = &m.meshes[b.mesh as usize];
            for p in 0..ms.count as usize {
                if m.derived()[bi].role[p] != Role::Dipole {
                    continue;
                }
                // a sheet in a symmetry plane of its mesh would coincide with its own image (as in
                // the low order)
                let sh = &ms.shapes[p];
                let tol = COINCIDENCE_TOL * sh.size;
                if (ms.isx != 0 && sh.centre[0].abs() <= tol && sh.normal[0].abs() > 0.5)
                    || (ms.isy != 0 && sh.centre[1].abs() <= tol && sh.normal[1].abs() > 0.5)
                {
                    return "a thin sheet in a symmetry plane of its mesh is not available".into();
                }
                if ms.geometry == PatchKind::Flat {
                    continue;
                }
                // a planar patch with curved sides: its points in the plane of its centre
                let sh = &ms.shapes[p];
                let mut off: f64 = 0.0;
                for q in &sh.hull {
                    off = off.max(dot(*q - sh.centre, sh.normal).abs());
                }
                if off > COINCIDENCE_TOL * sh.size {
                    return "higher-order thin sheets must be planar patches (curved dipole surfaces are not \
                            available)"
                        .into();
                }
            }
        }
        for p in &b.pressure {
            if p.mesh.is_some() {
                return "higher-order external pressure meshes are not available".into();
            }
            if m.has_infinite() {
                let sh = &m.meshes[b.mesh as usize].shapes[(p.panels[0].first - 1) as usize];
                if (sh.centre[2] + b.position[2]).abs() > COINCIDENCE_TOL * sh.size {
                    return "the omega = inf limit with a submerged pressure surface is not available".into();
                }
            }
        }
    }
    String::new()
}
