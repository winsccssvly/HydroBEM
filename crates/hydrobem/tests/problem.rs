//! The tests of the group "problem": the panel, reflection, Gauss and B-spline cases, the meshes (low order, flat and spline patches, their validation messages), the element
//! quadrature, the hydrostatics of a box against the closed form, and a free-surface plate (the
//! piston at omega = 0, its rules and the classification of points by it).

use hydrobem_core::internals::bspline::{basis_and_derivative, find_span};
use hydrobem_core::internals::geometry::{Panel, Vec3, make_panel, norm, reflect_vertices, reflections, v3};
use hydrobem_core::internals::inspect::{Scope, inspect_panels};
use hydrobem_core::internals::mesh::{Range, build_mesh, build_patch_mesh, check_knots, patch_eval};
use hydrobem_core::internals::mesh_desc::PatchKind;
use hydrobem_core::internals::mesh_desc::*;
use hydrobem_core::internals::model_codes::{InteriorMethod, Mass};
use hydrobem_core::internals::problem::{Body, Plate, Problem};
use hydrobem_core::internals::progress::SolveContext;
use hydrobem_core::internals::quadrature::{GAUSS_CACHED, gauss_legendre, gauss_nodes, gauss_rule, record_quadrature};
use hydrobem_core::internals::solver::solve;
use hydrobem_core::{ErrorKind, PanelRole, PointStatus, QuantityId};

// ---------------------------------------------------------------- panels, reflections, Gauss rules, B-splines

#[test]
fn panel_square() {
    let v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let mut p = Panel::default();
    assert!(make_panel(&v, &mut p).is_empty(), "unit square is valid");
    assert!((p.area - 1.0).abs() < 1e-15, "unit square area");
    assert!((p.normal[2] - 1.0).abs() < 1e-15, "counter-clockwise normal is +z");
    assert!((p.centroid[0] - 0.5).abs() < 1e-15 && (p.centroid[1] - 0.5).abs() < 1e-15, "square centroid");
}

#[test]
fn panel_triangle() {
    let v = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0]; // repeated last vertex
    let mut p = Panel::default();
    assert!(make_panel(&v, &mut p).is_empty(), "triangle by a repeated vertex is valid");
    assert_eq!(p.distinct, 3, "triangle has three distinct vertices");
    assert!((p.area - 2.0).abs() < 1e-15, "triangle area");
    assert!((p.centroid[0] - 2.0 / 3.0).abs() < 1e-15, "triangle centroid");
}

#[test]
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn panel_warped() {
    // non-planar quadrilateral: vertices alternately at z = +-h
    let h = 0.05;
    let v = [0.0, 0.0, h, 1.0, 0.0, -h, 1.0, 1.0, h, 0.0, 1.0, -h];
    let mut p = Panel::default();
    assert!(make_panel(&v, &mut p).is_empty(), "warped quad is valid after projection");
    // projection plane through the side midpoints is z = 0 here
    for i in 0..4 {
        assert!(p.proj[i][2].abs() < 1e-15, "warped quad projected to z = 0");
    }
    assert!((p.area - 1.0).abs() < 1e-14, "projected area");
}

#[test]
fn panel_failures() {
    let mut p = Panel::default();
    let line = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
    assert!(!make_panel(&line, &mut p).is_empty(), "collinear vertices rejected");
    let point = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(make_panel(&point, &mut p), "fewer than three distinct vertices", "two distinct vertices rejected");
    let bowtie = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert!(!make_panel(&bowtie, &mut p).is_empty(), "self-intersecting quad rejected");
    let concave = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 2.0, 0.0];
    assert!(!make_panel(&concave, &mut p).is_empty(), "non-convex quad rejected");
    let nan = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, f64::NAN, 0.0];
    assert_eq!(make_panel(&nan, &mut p), "non-finite vertex coordinate", "non-finite vertex rejected");
}

#[test]
fn reflection() {
    let v = [1.0, 0.0, -1.0, 2.0, 0.0, -1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // normal -y
    let mut p = Panel::default();
    assert!(make_panel(&v, &mut p).is_empty(), "side panel valid");
    for r in reflections(1, 1) {
        let w = reflect_vertices(&p.raw, &r);
        let mut buf = [0.0; 12];
        for i in 0..4 {
            for c in 0..3 {
                buf[3 * i + c] = w[i][c];
            }
        }
        let mut q = Panel::default();
        assert!(make_panel(&buf, &mut q).is_empty(), "reflected panel valid");
        assert!(
            (q.normal[1] - r.sy * p.normal[1]).abs() < 1e-15 && q.normal[0].abs() < 1e-15,
            "reflected normal is the reflection of the normal"
        );
    }
}

#[test]
fn gauss() {
    // the cache holds the rules of gauss_legendre, orders 1..16 (their polynomial exactness is the
    // unit test of quadrature.rs), and gauss_rule takes them from it
    for n in 1..=GAUSS_CACHED {
        let (x, w) = gauss_legendre(n);
        assert_eq!(gauss_nodes(n, false), &x[..]);
        assert_eq!(gauss_nodes(n, true), &w[..]);
        let (xr, wr) = gauss_rule(n);
        assert_eq!((&xr[..], &wr[..]), (&x[..], &w[..]));
    }
    // beyond the cache gauss_rule computes the rule: exact for polynomials of degree 2n - 1
    for n in [17, 24] {
        let (x, w) = gauss_rule(n);
        for d in 0..2 * n {
            let s: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| wi * xi.powi(d)).sum();
            let exact = if d % 2 == 1 { 0.0 } else { 2.0 / (d as f64 + 1.0) };
            assert!((s - exact).abs() < 1e-13, "Gauss-Legendre n={n} degree {d}: {s} vs {exact}");
        }
    }
}

#[test]
#[should_panic(expected = "gauss_nodes: order 17 outside 1..16")]
fn gauss_nodes_beyond_the_cache() {
    gauss_nodes(17, false); // a bug of the caller: the orders are fixed by the rules (1..16)
}

#[test]
fn bspline_partition() {
    let t = [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]; // cubic, 6 coefficients
    for u in [0.0, 0.1, 0.3, 0.45, 0.99, 1.0] {
        let span = find_span(&t, 6, 4, u);
        let mut b = [0.0; 8];
        basis_and_derivative(&t, 4, span, u, &mut b);
        let (mut s, mut ds) = (0.0, 0.0);
        for i in 0..4 {
            s += b[i];
            ds += b[4 + i];
        }
        assert!((s - 1.0).abs() < 1e-14, "B-spline partition of unity");
        assert!(ds.abs() < 1e-12, "B-spline derivative sums to zero");
        // derivative against a central difference
        let h = 1e-6;
        if u > h && u < 1.0 - h {
            let (mut bp, mut bm) = ([0.0; 8], [0.0; 8]);
            basis_and_derivative(&t, 4, find_span(&t, 6, 4, u + h), u + h, &mut bp);
            basis_and_derivative(&t, 4, find_span(&t, 6, 4, u - h), u - h, &mut bm);
            if find_span(&t, 6, 4, u + h) == span && find_span(&t, 6, 4, u - h) == span {
                for i in 0..4 {
                    assert!(((bp[i] - bm[i]) / (2.0 * h) - b[4 + i]).abs() < 1e-6, "B-spline derivative");
                }
            }
        }
    }
    // the spans: t[span] <= u < t[span + 1], the right end in the last span
    assert_eq!(find_span(&t, 6, 4, 0.0), 3);
    assert_eq!(find_span(&t, 6, 4, 0.3), 4);
    assert_eq!(find_span(&t, 6, 4, 0.45), 4);
    assert_eq!(find_span(&t, 6, 4, 0.6), 5);
    assert_eq!(find_span(&t, 6, 4, 1.0), 5);
}

// ---------------------------------------------------------------- meshes

#[test]
fn knot_checks() {
    assert_eq!(check_knots(&[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 5, 4, "u"), "");
    assert_eq!(check_knots(&[0.0, 0.0, 1.0, f64::NAN], 2, 2, "u"), "u knot vector has a non-finite value");
    assert_eq!(check_knots(&[0.0, 0.0, 1.0, 0.5], 2, 2, "v"), "v knot vector is not non-decreasing");
    assert_eq!(check_knots(&[1.0, 1.0, 1.0, 1.0], 2, 2, "u"), "u knot vector has zero length");
    assert_eq!(
        check_knots(&[0.0, 0.0, 0.1, 0.5, 1.0, 1.0, 1.0], 4, 3, "u"),
        "u knot vector is not complete (end knots must have multiplicity equal to the order)"
    );
    assert_eq!(
        check_knots(&[0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], 7, 4, "v"),
        "v knot vector has a C0 (or discontinuous) interior knot"
    );
}

/// The panels of a face p + s e1 + t e2 (s, t in [0, 1]), n1 x n2 of them, normal e1 x e2.
fn face(v: &mut Vec<f64>, p: Vec3, e1: Vec3, e2: Vec3, n1: usize, n2: usize) {
    for j in 0..n2 {
        for i in 0..n1 {
            let at = |a: usize, b: usize| p + (a as f64 / n1 as f64) * e1 + (b as f64 / n2 as f64) * e2;
            for q in [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)] {
                v.extend_from_slice(&q.0);
            }
        }
    }
}

/// A box |x| <= a, |y| <= b, -t <= z <= 0, open at z = 0, normals out of the box (into the fluid);
/// n panels along every edge of every face.
fn box_vertices(a: f64, b: f64, t: f64, n: usize) -> Vec<f64> {
    let mut v = Vec::new();
    face(&mut v, v3(-a, -b, -t), v3(0.0, 2.0 * b, 0.0), v3(2.0 * a, 0.0, 0.0), n, n); // bottom
    face(&mut v, v3(a, -b, -t), v3(0.0, 2.0 * b, 0.0), v3(0.0, 0.0, t), n, n); // x = a
    face(&mut v, v3(-a, -b, -t), v3(0.0, 0.0, t), v3(0.0, 2.0 * b, 0.0), n, n); // x = -a
    face(&mut v, v3(-a, b, -t), v3(0.0, 0.0, t), v3(2.0 * a, 0.0, 0.0), n, n); // y = b
    face(&mut v, v3(-a, -b, -t), v3(2.0 * a, 0.0, 0.0), v3(0.0, 0.0, t), n, n); // y = -b
    v
}

#[test]
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn low_order_mesh() {
    let v = box_vertices(2.0, 1.0, 0.5, 2);
    let desc = LowOrderMeshDesc {
        panel_count: (v.len() / 12) as i32,
        vertices: v,
        reference_length: 1.0,
        gravity: 9.81,
        symmetry_x: 0,
        symmetry_y: 0,
    };
    let m = build_mesh(&desc).unwrap();
    assert_eq!(m.count, 20);
    assert!(!m.patch);
    assert_eq!(m.shapes.len(), 20);
    // the bottom panels: normal -z, a quarter of the bottom each
    assert!((m.shapes[0].normal[2] + 1.0).abs() < 1e-15);
    assert!((m.shapes[0].area - 2.0).abs() < 1e-14);
    assert_eq!(m.shapes[0].hull.len(), 4);
    // the one-point rule of a panel: the centroid and n A
    let q = record_quadrature(&m, 0, &reflections(0, 0)[0]);
    assert_eq!(q.len(), 1);
    assert!((q[0].nds[2] + 2.0).abs() < 1e-14 && (q[0].x[2] + 0.5).abs() < 1e-15);

    // validation: the element of an invalid panel, the header
    let mut bad = desc.clone();
    for k in 12..24 {
        bad.vertices[k] = 0.0;
    }
    let e = build_mesh(&bad).expect_err("a degenerate panel");
    assert_eq!(e.kind(), ErrorKind::Value);
    assert_eq!(e.message(), "mesh element 2: fewer than three distinct vertices");
    assert_eq!(e.location().element, Some(2));
    let mut bad = desc.clone();
    bad.reference_length = 0.0;
    let e = build_mesh(&bad).expect_err("reference length 0");
    assert_eq!(e.message(), "mesh reference_length (ULEN) must be a positive finite number");
    let mut bad = desc.clone();
    bad.symmetry_y = 2;
    let e = build_mesh(&bad).expect_err("symmetry flag 2");
    assert_eq!(e.message(), "mesh symmetry flags (ISX, ISY) must be 0 or 1");
}

fn discretization(count: usize) -> Vec<PatchDiscretization> {
    vec![PatchDiscretization { nu: 2, nv: 2, ku: 3, kv: 3, iquo: 4, iqui: 4 }; count]
}

/// The flat patch mesh of the box (one patch per face).
fn box_patches(a: f64, b: f64, t: f64) -> PatchMeshDesc {
    let v = box_vertices(a, b, t, 1);
    let n = v.len() / 12;
    PatchMeshDesc {
        geometry: PatchKind::Flat,
        patch_count: n as i32,
        vertices: v,
        discretization: discretization(n),
        reference_length: 1.0,
        gravity: 9.81,
        ..PatchMeshDesc::default()
    }
}

/// One cubic B-spline patch of the plane z = -1 over [0, 1]^2 (the Greville abscissae as control
/// points: x = u, y = v), an interior knot at 0.5 in both directions.
fn spline_plane() -> PatchMeshDesc {
    let t = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let g = [0.0, 0.5 / 3.0, 0.5, 2.5 / 3.0, 1.0];
    let mut knots = t.to_vec();
    knots.extend_from_slice(&t);
    let mut c = Vec::new();
    for &y in &g {
        for &x in &g {
            c.extend_from_slice(&[x, y, -1.0]);
        }
    }
    PatchMeshDesc {
        geometry: PatchKind::Spline,
        patch_count: 1,
        spline_orders: vec![4, 4],
        spline_counts: vec![5, 5],
        knot_count: knots.len() as i64,
        knots,
        coefficient_count: 25,
        coefficients: c,
        discretization: discretization(1),
        reference_length: 1.0,
        gravity: 9.81,
        ..PatchMeshDesc::default()
    }
}

#[test]
fn patch_meshes() {
    // flat patches: the 3 x 3 Gauss rule integrates the area exactly
    let m = build_patch_mesh(&box_patches(2.0, 1.0, 0.5)).unwrap();
    assert!(m.patch && m.count == 5);
    let q = record_quadrature(&m, 0, &reflections(0, 0)[0]);
    assert_eq!(q.len(), 9);
    let nz: f64 = q.iter().map(|p| p.nds[2]).sum();
    assert!((nz + 8.0).abs() < 1e-13, "bottom n_z dS {nz}");

    // a spline patch: x = u, y = v on [0, 1]^2 at z = -1; normal into the fluid -z (x_u x x_v into the body)
    let m = build_patch_mesh(&spline_plane()).unwrap();
    assert!((m.shapes[0].normal[2] + 1.0).abs() < 1e-15);
    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    patch_eval(&m, 0, 0.2, -0.6, &mut x, &mut xu, &mut xv);
    assert!((x[0] - 0.6).abs() < 1e-14 && (x[1] - 0.2).abs() < 1e-14 && (x[2] + 1.0).abs() < 1e-15);
    assert!((xu[0] - 0.5).abs() < 1e-14 && xu[1].abs() < 1e-14 && (xv[1] - 0.5).abs() < 1e-14);
    // 2k = 8 points per span and direction, two spans each
    let q = record_quadrature(&m, 0, &reflections(0, 0)[0]);
    assert_eq!(q.len(), 4 * 64);
    let nz: f64 = q.iter().map(|p| p.nds[2]).sum();
    let mx: f64 = q.iter().map(|p| p.x[0] * p.x[0] * p.nds[2]).sum();
    assert!((nz + 1.0).abs() < 1e-14, "spline area {nz}");
    assert!((mx + 1.0 / 3.0).abs() < 1e-14, "spline x^2 moment {mx}");

    // validation messages name the patch
    let mut bad = spline_plane();
    bad.knots[4] = 0.0;
    bad.knots[5] = 0.0;
    let e = build_patch_mesh(&bad).expect_err("a knot vector of multiplicity 6 at 0");
    assert_eq!(
        e.message(),
        "patch 1: u knot vector is not complete (end knots must have multiplicity equal to the order)"
    );
    assert_eq!(e.location().element, Some(1));
    let mut bad = spline_plane();
    bad.coefficient_count = 24;
    let e = build_patch_mesh(&bad).expect_err("a wrong coefficient count");
    assert_eq!(e.message(), "spline coefficient count does not match the patch counts");
    let mut bad = box_patches(2.0, 1.0, 0.5);
    bad.discretization[2].ku = 7;
    let e = build_patch_mesh(&bad).expect_err("ku 7");
    assert_eq!(
        e.message(),
        "patch 3: the potential orders ku, kv (the columns 3, 4 of discretization) must be in 2..6"
    );
}

// ---------------------------------------------------------------- hydrostatics

/// A hydrostatics-only problem of the box, its rules checked.
fn box_problem(mesh: hydrobem_core::internals::mesh::Mesh, higher_order: bool) -> Problem {
    let mut p = Problem::default();
    p.meshes = vec![mesh];
    p.bodies.push(Body { name: "box".into(), mesh: 0, ..Body::default() });
    p.opt.radiation = false;
    p.opt.diffraction = false;
    p.opt.higher_order = higher_order;
    p.check_rules().unwrap();
    p
}

/// The hydrostatics of the box against the closed form; tol relative to the largest stiffness.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
// The hydrostatics of the box |x| <= a, |y| <= b, -t <= z <= 0 against its closed forms; panel: the
// sizes (h_x, h_y) of the bottom panels of a low-order mesh, whose one-point rule misses their own
// second moments, A h^2 / 12 of the water-plane inertia in each direction (exact for patches: None).
fn check_box(p: &Problem, a: f64, b: f64, t: f64, panel: Option<(f64, f64)>) {
    let h = p.hydrostatics(0);
    let (rho, g) = (p.env.density, p.env.gravity);
    let vol = 4.0 * a * b * t;
    for k in 0..3 {
        assert!((h.volumes[k] - vol).abs() < 1e-12 * vol, "volume {k}: {}", h.volumes[k]);
    }
    assert!(h.cb[0].abs() < 1e-12 && h.cb[1].abs() < 1e-12 && (h.cb[2] + 0.5 * t).abs() < 1e-12);
    assert!((h.wp_area - 4.0 * a * b).abs() < 1e-12);
    assert!(h.wp_centroid[0].abs() < 1e-12 && h.wp_centroid[1].abs() < 1e-12);
    let mut c = [0.0; 36];
    c[2 * 6 + 2] = rho * g * 4.0 * a * b;
    c[3 * 6 + 3] = rho * g * ((2.0 * a) * (2.0 * b).powi(3) / 12.0 - vol * 0.5 * t);
    c[4 * 6 + 4] = rho * g * ((2.0 * b) * (2.0 * a).powi(3) / 12.0 - vol * 0.5 * t);
    if let Some((hx, hy)) = panel {
        let area = 4.0 * a * b;
        c[3 * 6 + 3] -= rho * g * area * hy * hy / 12.0;
        c[4 * 6 + 4] -= rho * g * area * hx * hx / 12.0;
    }
    let scale = c[4 * 6 + 4].abs();
    for k in 0..36 {
        assert!((h.c_rho_g[k] - c[k]).abs() <= 1e-12 * scale, "C[{k}] {} vs {}", h.c_rho_g[k], c[k]);
        assert_eq!(h.c[k], h.c_rho_g[k], "no weight without a mass");
    }
    assert!((h.c_rho_g[2 * 6 + 2] - c[2 * 6 + 2]).abs() < 1e-12 * c[2 * 6 + 2], "C33 exact");
    assert!(h.mass.is_nan() && !h.has_weight && h.cg[0].is_nan());
}

#[test]
fn box_hydrostatics_low_order() {
    let (a, b, t) = (2.0, 1.0, 0.5);
    let n = 20;
    let v = box_vertices(a, b, t, n);
    let desc = LowOrderMeshDesc {
        panel_count: (v.len() / 12) as i32,
        vertices: v,
        reference_length: 1.0,
        gravity: 9.81,
        symmetry_x: 0,
        symmetry_y: 0,
    };
    let p = box_problem(build_mesh(&desc).unwrap(), false);
    assert!(p.warnings.is_empty(), "a closed box has no mesh warnings: {:?}", p.warnings);
    // the discrete hydrostatics of the one-point rule, exact to round-off
    check_box(&p, a, b, t, Some((2.0 * a / n as f64, 2.0 * b / n as f64)));
}

#[test]
fn box_hydrostatics_patches() {
    let (a, b, t) = (2.0, 1.0, 0.5);
    let p = box_problem(build_patch_mesh(&box_patches(a, b, t)).unwrap(), true);
    assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    check_box(&p, a, b, t, None);
}

#[test]
fn box_hydrostatics_symmetric_with_weight() {
    // the quarter x, y >= 0 stored with both symmetry flags, a mass and a centre of gravity
    let (a, b, t) = (2.0, 1.0, 0.5);
    let full = box_patches(a, b, t);
    let mut v = Vec::new();
    face(&mut v, v3(0.0, 0.0, -t), v3(0.0, b, 0.0), v3(a, 0.0, 0.0), 1, 1); // bottom
    face(&mut v, v3(a, 0.0, -t), v3(0.0, b, 0.0), v3(0.0, 0.0, t), 1, 1); // x = a
    face(&mut v, v3(0.0, b, -t), v3(0.0, 0.0, t), v3(a, 0.0, 0.0), 1, 1); // y = b
    let quarter = PatchMeshDesc {
        patch_count: 3,
        vertices: v,
        discretization: discretization(3),
        symmetry_x: 1,
        symmetry_y: 1,
        ..full
    };
    let mut p = Problem::default();
    p.meshes = vec![build_patch_mesh(&quarter).unwrap()];
    let zg = -0.1;
    p.bodies.push(Body {
        name: "box".into(),
        mesh: 0,
        has_cg: true,
        cg: v3(0.0, 0.0, zg),
        mass: Some(Mass::Displaced),
        ..Body::default()
    });
    p.opt.radiation = false;
    p.opt.diffraction = false;
    p.opt.higher_order = true;
    p.check_rules().unwrap();
    let h = p.hydrostatics(0);
    let vol = 4.0 * a * b * t;
    assert!((h.volumes[2] - vol).abs() < 1e-12 * vol);
    let mass = p.env.density * vol;
    assert!((h.mass - mass).abs() < 1e-9 * mass && h.has_weight);
    let mg = mass * p.env.gravity;
    let c44 = p.env.density * p.env.gravity * ((2.0 * a) * (2.0 * b).powi(3) / 12.0 - vol * 0.5 * t) - mg * zg;
    assert!((h.c[3 * 6 + 3] - c44).abs() < 1e-9 * c44.abs(), "C44 with the weight {} vs {c44}", h.c[3 * 6 + 3]);
    assert!((h.cg[2] - zg).abs() < 1e-15);
}

#[test]
fn rule_messages() {
    let mut p = Problem::default();
    let e = p.check_rules().expect_err("no bodies");
    assert_eq!(e.message(), "the model has no bodies");

    let mesh = build_patch_mesh(&box_patches(2.0, 1.0, 0.5)).unwrap();
    let mut p = Problem::default();
    p.meshes = vec![mesh];
    p.bodies.push(Body { name: "box".into(), mesh: 0, ..Body::default() });
    p.opt.radiation = false;
    p.opt.diffraction = false;
    let e = p.check_rules().expect_err("a patch mesh in the low order");
    assert_eq!(e.message(), "bodies/box: a patch mesh requires options.higher_order = true");
    assert_eq!(e.location().body, Some(0));
    p.opt.higher_order = true;
    p.bodies[0].dipole_panels = vec![hydrobem_core::internals::mesh::Range { first: 4, last: 6 }];
    let e = p.check_rules().expect_err("a range outside the mesh");
    assert_eq!(e.message(), "bodies/box dipoles: panel range 4-6 is outside the mesh (elements 1-5)");
    p.bodies[0].dipole_panels.clear();
    p.omegas = vec![1.0, 1.0];
    p.opt.radiation = true;
    p.bodies[0].modes = vec![3];
    let e = p.check_rules().expect_err("a duplicate frequency");
    assert_eq!(e.message(), "environment.omegas has the duplicate value 1");
}

// ---------------------------------------------------------------- interior free surfaces

#[test]
fn interior_surfaces_of_a_box() {
    use hydrobem_core::internals::interior::interior_panels;

    let (a, b, t) = (2.0, 1.0, 0.5);
    let full = box_vertices(a, b, t, 4);
    let mut quarter = Vec::new();
    face(&mut quarter, v3(0.0, 0.0, -t), v3(0.0, b, 0.0), v3(a, 0.0, 0.0), 3, 3);
    face(&mut quarter, v3(a, 0.0, -t), v3(0.0, b, 0.0), v3(0.0, 0.0, t), 3, 2);
    face(&mut quarter, v3(0.0, b, -t), v3(0.0, 0.0, t), v3(a, 0.0, 0.0), 2, 3);
    for (v, sym) in [(full, 0), (quarter, 1)] {
        let desc = LowOrderMeshDesc {
            panel_count: (v.len() / 12) as i32,
            vertices: v,
            reference_length: 1.0,
            gravity: 9.81,
            symmetry_x: sym,
            symmetry_y: sym,
        };
        for method in [InteriorMethod::Projected, InteriorMethod::Waterline, InteriorMethod::Graded] {
            let mut p = Problem::default();
            p.meshes = vec![build_mesh(&desc).unwrap()];
            p.bodies.push(Body { name: "box".into(), mesh: 0, interior_method: method, ..Body::default() });
            p.opt.radiation = false;
            p.opt.diffraction = false;
            p.check_rules().unwrap();
            let ip = interior_panels(&p, 0).unwrap();
            assert_eq!((ip.isx, ip.isy), (sym, sym));
            let stored: f64 = ip.panels.iter().map(|q| q.area).sum();
            let copies = if sym == 1 { 4.0 } else { 1.0 };
            assert!((stored * copies - 4.0 * a * b).abs() < 1e-12, "method {method:?} sym {sym}: area {stored}");
            for q in &ip.panels {
                assert!((q.normal[2] - 1.0).abs() < 1e-12 && q.centroid[2].abs() < 1e-15);
            }
            let global = inspect_panels(&p, Scope::Body(0), &[PanelRole::InteriorSurface], true).unwrap();
            assert_eq!(global.len(), ip.panels.len() * copies as usize);
        }
    }
}

/// The side of the cylinder of radius 1 and draft 0.5 (quadrant) as a rational patch: the
/// quarter circle of order 3 with the middle weight cos 45° (Piegl & Tiller 1997, §7.3), linear
/// in depth; `scale` multiplies every weight, `w` gives the weights (None: the circle's).
fn rational_side(scale: f64, w: Option<[f64; 3]>) -> PatchMeshDesc {
    let wq = w.unwrap_or([1.0, 0.5f64.sqrt(), 1.0]);
    let mut knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    knots.extend_from_slice(&[0.0, 0.0, 1.0, 1.0]);
    let mut c = Vec::new();
    for z in [0.0, -0.5] {
        for (x, y) in [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            c.extend_from_slice(&[x, y, z]);
        }
    }
    PatchMeshDesc {
        geometry: PatchKind::Spline,
        patch_count: 1,
        spline_orders: vec![3, 2],
        spline_counts: vec![3, 2],
        knot_count: knots.len() as i64,
        knots,
        coefficient_count: 6,
        coefficients: c,
        weights: [wq, wq].concat().iter().map(|w| scale * w).collect(),
        discretization: discretization(1),
        reference_length: 1.0,
        gravity: 9.81,
        ..PatchMeshDesc::default()
    }
}

#[test]
fn rational_patches_are_exact_and_their_derivatives_differences() {
    // the rational side lies on the circle; its derivatives are the central differences of its
    // points; both evaluators (the mesh and the higher-order patch geometry) agree; a common
    // factor of the weights leaves the surface; all weights 1 is the non-rational patch
    let m = build_patch_mesh(&rational_side(1.0, None)).unwrap();
    let scaled = build_patch_mesh(&rational_side(3.7, None)).unwrap();
    let ones = build_patch_mesh(&rational_side(1.0, Some([1.0, 1.0, 1.0]))).unwrap();
    let mut plain = rational_side(1.0, None);
    plain.weights.clear();
    let plain = build_patch_mesh(&plain).unwrap();
    let g = hydrobem_core::internals::ho_geometry::patch_geo(&m, 0);
    let eval = |m: &hydrobem_core::internals::mesh::Mesh, u: f64, v: f64| {
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        patch_eval(m, 0, u, v, &mut x, &mut xu, &mut xv);
        (x, xu, xv)
    };
    let d = 1e-6;
    let (mut circle, mut diff, mut twin, mut scale, mut one) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..9 {
        for j in 0..5 {
            let (u, v) = (-0.9 + 0.225 * i as f64, -0.8 + 0.4 * j as f64);
            let (x, xu, xv) = eval(&m, u, v);
            circle = circle.max((x[0].hypot(x[1]) - 1.0).abs());
            let fu = (0.5 / d) * (eval(&m, u + d, v).0 - eval(&m, u - d, v).0);
            let fv = (0.5 / d) * (eval(&m, u, v + d).0 - eval(&m, u, v - d).0);
            diff = diff.max(norm(fu - xu)).max(norm(fv - xv));
            let (mut y, mut yu, mut yv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            g.eval(u, v, &mut y, &mut yu, &mut yv);
            twin = twin.max(norm(y - x)).max(norm(yu - xu)).max(norm(yv - xv));
            let (s, su, sv) = eval(&scaled, u, v);
            scale = scale.max(norm(s - x)).max(norm(su - xu)).max(norm(sv - xv));
            let ((a, au, av), (b, bu, bv)) = (eval(&ones, u, v), eval(&plain, u, v));
            one = one.max(norm(a - b)).max(norm(au - bu)).max(norm(av - bv));
        }
    }
    println!(
        "rational: circle {circle:.1e}, differences {diff:.1e}, evaluators {twin:.1e}, scale {scale:.1e}, ones {one:.1e}"
    );
    assert!(circle < 1e-15 && diff < 1e-8 && twin < 1e-15 && scale < 1e-14 && one < 1e-15);
    // a weight must be positive
    let e = build_patch_mesh(&rational_side(1.0, Some([1.0, 0.0, 1.0]))).unwrap_err();
    assert!(e.message().contains("positive"), "{}", e.message());
}

// ---------------------------------------------------------------- free-surface plates

/// The quadrant of a disc of radius a in z = 0 (n_r uniform rings, n_phi divisions), a low-order
/// mesh with the symmetry of both planes when sym = 1.
fn quarter_disc(a: f64, n_r: usize, n_phi: usize, sym: i32) -> hydrobem_core::internals::mesh::Mesh {
    let mut v = Vec::new();
    let p = |r: f64, k: usize| {
        let t = 0.5 * std::f64::consts::PI * k as f64 / n_phi as f64;
        [r * t.cos(), r * t.sin(), 0.0]
    };
    for i in 0..n_phi {
        for j in 0..n_r {
            let (r0, r1) = (a * j as f64 / n_r as f64, a * (j + 1) as f64 / n_r as f64);
            for q in [p(r0, i), p(r0, i + 1), p(r1, i + 1), p(r1, i)] {
                v.extend_from_slice(&q);
            }
        }
    }
    let desc = LowOrderMeshDesc {
        panel_count: (v.len() / 12) as i32,
        vertices: v,
        reference_length: 1.0,
        gravity: 9.81,
        symmetry_x: sym,
        symmetry_y: sym,
    };
    build_mesh(&desc).unwrap()
}

/// A body of a plate only: the elements 1..elements of its mesh, heaving.
fn plate_body(name: &str, mesh: i32, records: i32, position: Vec3) -> Body {
    let plate = Plate { panels: vec![Range { first: 1, last: records }], ..Plate::default() };
    Body { name: name.into(), mesh, modes: vec![3], position, plates: vec![plate], ..Body::default() }
}

#[test]
fn a_free_surface_plate() {
    // theory §9.7: at omega = 0 a disc of radius a in z = 0 heaving on deep water is a piston in the
    // rigid plane, A33(0) = (8/3) rho a^3 (Rayleigh's end correction 8a/(3 pi) on the area pi a^2);
    // 8 uniform rings a quadrant: 6.0e-4 (the Python tests: 6.4e-5 on 16)
    let (n_r, n_phi) = (8, 16);
    let mut p = Problem::default();
    p.meshes = vec![quarter_disc(1.0, n_r, n_phi, 1)];
    p.bodies.push(plate_body("plate", 0, (n_r * n_phi) as i32, Vec3::ZERO));
    p.omegas = vec![0.0];
    p.opt.diffraction = false;
    p.check_rules().unwrap();
    assert_eq!(p.derived()[0].plate, (n_r * n_phi) as i32);
    let r = solve(&mut p, &SolveContext::default()).unwrap();
    let a0 = r.f32s(QuantityId::AddedMassZeroFrequency).unwrap()[2] as f64; // [D = 6, K = 1]: heave row
    let err = a0 / (8.0 / 3.0 * p.env.density) - 1.0;
    println!("free-surface plate: A33(0) of the disc against (8/3) rho a^3: {err:.1e}");
    assert!(err.abs() < 1e-3, "A33(0) {a0}: {err:.1e}");

    // the classification of points by the plate: on it inside its waterplane, just under it near the
    // body, beside it in the fluid
    let pts = [v3(0.3, 0.2, 0.0), v3(0.3, 0.2, -0.05), v3(3.0, 0.0, 0.0)];
    let s: Vec<PointStatus> = p.classify_field_points(&pts, &[]).iter().map(|c| c.status).collect();
    assert_eq!(s, [PointStatus::InsideWaterplane, PointStatus::NearBody, PointStatus::Fluid]);

    // a second plate body over the first (the quadrant moved by 0.5) overlaps it: refused; moved by
    // 1.5 (apart) accepted
    for (x, ok) in [(0.5, false), (1.5, true)] {
        let mut q = Problem::default();
        q.meshes = vec![quarter_disc(1.0, 4, 8, 0), quarter_disc(1.0, 4, 8, 0)];
        q.bodies.push(plate_body("a", 0, 32, Vec3::ZERO));
        q.bodies.push(plate_body("b", 1, 32, v3(x, 0.0, 0.0)));
        q.omegas = vec![1.0];
        q.opt.diffraction = false;
        let e = q.check_rules();
        if ok {
            e.unwrap();
        } else {
            let m = e.unwrap_err().message().to_string();
            assert!(m.contains("free-surface plate 1 overlaps bodies/"), "{m}");
        }
    }
}
