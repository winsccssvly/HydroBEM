//! Tests of the group "prep": preprocessing (waterline, trim, automatic control surface), the
//! inspection, the response (Kochin angles, momentum drift, mode displacements), the drift
//! integrators, the memory estimate, the routes and the motions.

use hydrobem_core::ErrorKind;
use hydrobem_core::internals::mesh_desc::{AnalyticKind, PatchKind};
use hydrobem_core::internals::model_codes::{Domain, WaterlineTerm};
use num_complex::{Complex32, Complex64};

use hydrobem_core::internals::analytic::AnalyticPatch;
use hydrobem_core::internals::control_drift::ControlSurfaceDrift;
use hydrobem_core::internals::geometry::{Panel, make_panel, v3};
use hydrobem_core::internals::inspect::inspect_patch_points;
use hydrobem_core::internals::low_order::{Column, Output, Plan};
use hydrobem_core::internals::mesh::Mesh;
use hydrobem_core::internals::mesh_desc::*;
use hydrobem_core::internals::motion::solve_motions;
use hydrobem_core::internals::preprocess::{automatic_control_surface, trim_low_order, waterline_segments};
use hydrobem_core::internals::pressure_drift::PressureDrift;
use hydrobem_core::internals::problem::{Body, Problem};
use hydrobem_core::internals::progress::SolveContext;
use hydrobem_core::internals::response::auto_kochin_angles;
use hydrobem_core::internals::route::{Method, Route, select_route};

fn panel(v: [[f64; 3]; 4]) -> Panel {
    let flat: Vec<f64> = v.iter().flatten().copied().collect();
    let mut p = Panel::default();
    let why = make_panel(&flat, &mut p);
    assert!(why.is_empty(), "{why}");
    p
}

fn low_order_mesh(panels: Vec<Panel>) -> Mesh {
    Mesh { patch: false, count: panels.len() as i32, panels, ..Mesh::default() }
}

/// The wetted part of a square column of half-width 1 and draft 1: four walls with their top edge
/// in z = 0 and the bottom (normals into the fluid).
fn square_column() -> Mesh {
    low_order_mesh(vec![
        panel([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]),
        panel([[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        panel([[-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]),
        panel([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]]),
        panel([[-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]),
    ])
}

// The body waterline of a low-order mesh: the four top edges of the walls, the bottom
// left out; nothing in a plane the mesh does not reach.
#[test]
fn waterline_of_square_column() {
    let m = square_column();
    let segs = waterline_segments(&m, 0.0, 1e-6);
    assert_eq!(segs.len(), 4);
    for s in &segs {
        assert!(s[0][2] == 0.0 && s[1][2] == 0.0);
        let l = ((s[1][0] - s[0][0]).powi(2) + (s[1][1] - s[0][1]).powi(2)).sqrt();
        assert!((l - 2.0).abs() < 1e-14);
    }
    assert!(waterline_segments(&m, 0.5, 1e-6).is_empty());
    // the bottom at z = -1 is wholly in that plane: its edges are not a waterline, those of the walls are
    assert_eq!(waterline_segments(&m, -1.0, 1e-6).len(), 4);
}

// The automatic control surface about the column: the cylinder side
// and bottom, and one segment-arc patch per waterline chord; the panel counts by arc length.
#[test]
fn automatic_control_surface_circle() {
    let m = square_column();
    let d = ControlSurfaceDesc {
        radius: 5.0,
        depth: 3.0,
        panel_size: 1.0,
        waterline_z: 0.0,
        free_surface: 1,
        ..Default::default()
    };
    let cs = automatic_control_surface(&m, &d).unwrap();
    assert_eq!(cs.patches.len(), 6);
    assert_eq!(cs.discretization.len(), 6);
    assert_eq!(cs.patches[0].kind, AnalyticKind::Line);
    assert_eq!(cs.patches[1].kind, AnalyticKind::Line);
    for p in &cs.patches[2..] {
        assert_eq!(p.kind, AnalyticKind::SegmentArc);
    }
    // the side: 2 pi 5 around (32 panels of 1), 3 down (4); the bottom: 5 to the axis (6)
    assert_eq!((cs.discretization[0].nu, cs.discretization[0].nv), (32, 4));
    assert_eq!((cs.discretization[1].nu, cs.discretization[1].nv), (32, 6));
    // the free-surface patches cover the azimuths once
    let span: f64 = cs.patches[2..].iter().map(|p: &AnalyticPatch| (p.theta1 - p.theta0).abs()).sum();
    assert!((span - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    // the errors of the description
    let bad = ControlSurfaceDesc { depth: 0.0, ..d.clone() };
    let e = automatic_control_surface(&m, &bad).err().unwrap();
    assert_eq!(e.message(), "control surface: DEPTH must be positive and finite");
    let bad = ControlSurfaceDesc { radius: 0.0, ..d.clone() };
    let e = automatic_control_surface(&m, &bad).err().unwrap();
    assert_eq!(
        e.message(),
        "automatic control surface: give the radius of a circular outer surface or a boundary polygon"
    );
}

// The low-order trim: a pitch-free trim at an elevation of 0.5 keeps the lower half of the
// walls and the bottom.
#[test]
fn trim_low_order_column() {
    let m = square_column();
    let t = TrimDesc { elevation: 0.5, ..Default::default() };
    let lt = trim_low_order(&m, &t).unwrap();
    assert_eq!(lt.panels.len(), 5);
    for p in &lt.panels {
        for x in p {
            assert!(x[2] <= 0.0);
        }
    }
}

// The Kochin angle count of "auto": 8 ceil((2 ceil(k R) + 96) / 8).
#[test]
fn kochin_angles_auto() {
    let mut m = Problem::default();
    m.meshes.push(square_column());
    m.bodies.push(Body { mesh: 0, ..Body::default() });
    m.omegas = vec![0.0, 1.0, f64::INFINITY];
    // k = 1 / g, R = sqrt 2: ceil(k R) = 1, (2 + 96) -> 104
    assert_eq!(auto_kochin_angles(&m), 104);
    m.bodies[0].position = v3(100.0, 0.0, 0.0);
    // R = 100 + sqrt 2: ceil(k R) = 11, 118 -> 120
    assert_eq!(auto_kochin_angles(&m), 120);
    m.omegas = vec![0.0];
    assert_eq!(auto_kochin_angles(&m), 0);
}

// The route of a hydrostatics-only problem, and a reason text of the route rules.
#[test]
fn route_reasons() {
    let mut m = Problem::default();
    m.opt.radiation = false;
    m.opt.diffraction = false;
    assert_eq!(select_route(&m).unwrap(), Route::Hydrostatics);
    assert_eq!(select_route(&m).unwrap().name(), "hydrostatics");
    m.opt.radiation = true;
    m.opt.drift_angles = 8;
    m.opt.domain = Domain::XWall;
    let e = select_route(&m).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::Unsupported);
    assert_eq!(
        e.message(),
        "no solver route for this problem in this build: the mean drift with walls or a channel is not available"
    );
}

// The names of the routes, as the results' meta data and the .hbr give them:
// <method>_<infinite|finite>_<domain>, and "hydrostatics".
#[test]
fn route_names() {
    let domains = [
        (Domain::OpenSea, "open_sea"),
        (Domain::XWall, "x_wall"),
        (Domain::YWall, "y_wall"),
        (Domain::OrthogonalWalls, "orthogonal_walls"),
        (Domain::Channel, "channel"),
    ];
    for (method, m) in [(Method::LowOrder, "low_order"), (Method::HigherOrder, "higher_order")] {
        for (finite_depth, depth) in [(false, "infinite"), (true, "finite")] {
            for (domain, d) in domains {
                assert_eq!(Route::Solve { method, finite_depth, domain }.name(), format!("{m}_{depth}_{d}"));
            }
        }
    }
    assert_eq!(Route::Hydrostatics.name(), "hydrostatics");
    let mut m = Problem::default();
    m.omegas = vec![1.0];
    m.headings = vec![0.0];
    m.env.depth = 10.0;
    m.opt.domain = Domain::YWall;
    let r = select_route(&m).unwrap();
    assert_eq!(r, Route::Solve { method: Method::LowOrder, finite_depth: true, domain: Domain::YWall });
    assert_eq!(r.name(), "low_order_finite_y_wall");
}

// A route equals its name as every string type, on either side; outside the crate a match on the
// non-exhaustive enums has a wildcard arm.
#[test]
fn a_route_equals_its_name_as_any_string() {
    let r = Route::Solve { method: Method::HigherOrder, finite_depth: false, domain: Domain::Channel };
    let name = String::from("higher_order_infinite_channel");
    let s: &str = &name;
    assert!(r == s && s == r);
    assert!(r == *s && *s == r);
    assert!(r == name && name == r);
    let hydrostatics = String::from("hydrostatics");
    assert!(r != "hydrostatics" && r != hydrostatics && Route::Hydrostatics == hydrostatics);
    let method = match r {
        Route::Solve { method: Method::LowOrder, .. } => "low",
        Route::Solve { method: Method::HigherOrder, .. } => "higher",
        _ => "other",
    };
    assert_eq!(method, "higher");
}

// The points and normals of a flat patch (the bilinear map of its corners, x_u x x_v into the fluid).
#[test]
fn patch_points_flat() {
    let m = Mesh {
        patch: true,
        geometry: PatchKind::Flat,
        count: 1,
        vertices: vec![0.0, 0.0, -1.0, 2.0, 0.0, -1.0, 2.0, 2.0, -1.0, 0.0, 2.0, -1.0],
        ..Mesh::default()
    };
    let uv = [0.0, 0.0, -1.0, 1.0];
    let (pts, nrm) = inspect_patch_points(&m, 1, 2, &uv).unwrap();
    assert_eq!(pts, [1.0, 1.0, -1.0, 0.0, 2.0, -1.0]);
    assert_eq!(nrm, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let e = inspect_patch_points(&m, 2, 1, &uv).err().unwrap();
    assert_eq!(e.message(), "patch 2 does not exist (1 patches)");
    let e = inspect_patch_points(&m, 1, 1, &[1.5, 0.0]).err().unwrap();
    assert_eq!(e.message(), "patch parameters must be in [-1, 1]");
}

// The motions of one column: xi = X / (C - w^2 (M + A) + i w (B + B_ext)).
#[test]
fn motions_one_column() {
    let mut m = Problem::default();
    m.bodies.push(Body::default());
    let pl = Plan {
        rad: vec![Column { coord: 0, body: 0, mode: 3, ..Column::default() }],
        D: 1,
        rao: true,
        CC: vec![10.0],
        MM: vec![2.0],
        BB: vec![0.5],
        ..Plan::default()
    };
    let mut out = Output {
        A: vec![1.0],
        B: vec![0.25],
        X: vec![Complex32::new(3.0, -1.0)],
        M: vec![Complex32::new(0.0, 0.0)],
        ..Output::default()
    };
    let w = 1.5;
    solve_motions(&m, &pl, w, 1, &mut out, SolveContext::default().statics()).unwrap();
    let z = Complex64::new(10.0 - w * w * 3.0, w * 0.75);
    let x = Complex64::new(3.0, -1.0) / z;
    assert!((out.M[0].re as f64 - x.re).abs() < 1e-6 && (out.M[0].im as f64 - x.im).abs() < 1e-6);
    // a singular system: the message of the motion equation
    let pl0 = Plan { CC: vec![0.0], MM: vec![0.0], BB: vec![0.0], ..pl.clone() };
    let mut out0 = Output { A: vec![0.0], B: vec![0.0], ..out.clone() };
    let e = solve_motions(&m, &pl0, w, 1, &mut out0, SolveContext::default().statics()).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::Numerical);
    assert!(e.message().starts_with("motion equation at omega = 1.5: "), "{}", e.message());
}

// The pressure-integration drift (theory §11.2): the waterline term -rho g / 2 <zeta^2> n dl of a
// body at rest, and the moment about the moving origin.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn pressure_drift_waterline() {
    let (rho, g, w) = (1025.0, 9.81, 0.8);
    let X = [Complex64::new(0.0, 0.0); 6];
    let mut pd = PressureDrift::new(&X, &v3(0.0, 0.0, 0.0), rho, g, w, false);
    // zeta = -(i w / g) phi = 1
    let phi = Complex64::new(0.0, g / w);
    pd.waterline(phi, &v3(2.0, 0.0, 0.0), &v3(1.0, 0.0, 0.0), 0.5);
    let mut o = [0.0; 9];
    pd.finish(0.0, &[0.0; 36], &mut o);
    let f = -0.25 * rho * g * 0.5;
    assert!((o[0] - f).abs() < 1e-9 * f.abs());
    assert!(o[1] == 0.0 && o[2] == 0.0 && o[3] == 0.0 && o[4] == 0.0 && o[5] == 0.0);
    // the hydrostatic pressure of a point at rest: -p0 n weight
    let mut pd = PressureDrift::new(&X, &v3(0.0, 0.0, 0.0), rho, g, w, false);
    let v = [Complex64::new(0.0, 0.0); 3];
    pd.surface(Complex64::new(0.0, 0.0), &v, &v3(1.0, 0.0, -2.0), &v3(0.0, 0.0, 1.0), -2.0, 0.1);
    pd.finish(-2.0, &[0.0; 36], &mut o);
    assert!(o.iter().all(|x| *x == 0.0), "the mean drift of a body at rest is zero: {o:?}");
}

// The control-surface drift: the CL term -rho g / 2 <zeta^2> n' dl, none with
// alternative 2; the moment about the moving origin M - <xi_O x F1>.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn control_surface_drift_terms() {
    let (rho, g, w) = (1000.0, 9.80665, 1.2);
    let mut X = [Complex64::new(0.0, 0.0); 6];
    X[0] = Complex64::new(1.0, 0.0);
    let phi = Complex64::new(0.0, g / w); // zeta = 1
    let mut c = ControlSurfaceDrift::new(&X, &v3(0.0, 0.0, 0.0), rho, g, w, WaterlineTerm::ControlSurface);
    c.line(phi, &v3(3.0, 0.0, 0.0), &v3(1.0, 0.0, 0.0), 2.0);
    let mut o = [0.0; 9];
    c.finish(&[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)], &mut o);
    let s = -0.5 * rho * g * 0.5 * 2.0;
    assert!((o[0] - s).abs() < 1e-9 * s.abs());
    // xi_O x conj(F1) = (1, 0, 0) x (0, 1, 0) = (0, 0, 1)
    assert!((o[8] - (o[5] - 0.5)).abs() < 1e-12);
    let mut c2 = ControlSurfaceDrift::new(&X, &v3(0.0, 0.0, 0.0), rho, g, w, WaterlineTerm::Body);
    c2.line(phi, &v3(3.0, 0.0, 0.0), &v3(1.0, 0.0, 0.0), 2.0);
    c2.add(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    c2.finish(&[Complex64::new(0.0, 0.0); 3], &mut o);
    assert_eq!(o, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]);
}

#[test]
fn memory_estimate_hydrostatics() {
    let mut m = Problem::default();
    m.opt.radiation = false;
    m.opt.diffraction = false;
    // a problem without bodies is refused by the rules
    let r = hydrobem_core::internals::memory::memory_estimate(&mut m);
    let e = r.expect_err("a problem without bodies");
    assert_eq!((e.kind(), e.message()), (ErrorKind::Value, "the model has no bodies"));
}
