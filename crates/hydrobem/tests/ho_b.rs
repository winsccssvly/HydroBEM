//! Unit tests of the helpers of one higher-order frequency: the L2 projection (ho_geometry.rs,
//! ho_internal.rs), the spline values and the inline code of ho_internal.rs. The frequency itself (solve_frequency) and the solve
//! (solve_higher_order) are tested end to end by the regression and Python suites.

use hydrobem_core::internals::geometry::{Reflection, v3};
use hydrobem_core::internals::ho_geometry::{build_projections, projection_bytes};
use hydrobem_core::internals::ho_internal::{
    HoGeom, Inst, PatchDisc, Pt, mode_index, mode_sign, project, spline_values, surface_gradient,
};
use hydrobem_core::internals::low_order::Column;
use hydrobem_core::internals::model_codes::PressureShape;
use num_complex::Complex64;

/// One instance of one bilinear patch (one panel, orders 2 x 2: four coefficients, the values at
/// the corners) with the 2 x 2 Gauss points as its inner points (exact for the mass matrix).
fn bilinear_patch() -> HoGeom<'static> {
    let mut h = HoGeom {
        disc: vec![PatchDisc {
            nu: 1,
            nv: 1,
            ku: 2,
            kv: 2,
            iquo: 2,
            iqui: 2,
            Tu: vec![-1.0, -1.0, 1.0, 1.0],
            Tv: vec![-1.0, -1.0, 1.0, 1.0],
        }],
        inst: vec![Inst::default()],
        col0: vec![0],
        dof0: vec![0],
        blk0: vec![0],
        Ncol: 4,
        Ndof: 4,
        ..HoGeom::default()
    };
    let g = 1.0 / 3.0f64.sqrt();
    for &v in &[-g, g] {
        for &u in &[-g, g] {
            let mut p = Pt { uf: u, vf: v, w: 1.0, ..Pt::default() };
            p.bu[0] = 0.5 * (1.0 - u);
            p.bu[1] = 0.5 * (1.0 + u);
            p.bv[0] = 0.5 * (1.0 - v);
            p.bv[1] = 0.5 * (1.0 + v);
            h.inner.push(p);
        }
    }
    h.inner0 = vec![0, h.inner.len() as i32];
    h
}

fn bilinear(u: f64, v: f64) -> Complex64 {
    Complex64::new(1.0 + 2.0 * u - 0.5 * v + 0.25 * u * v, -0.5 + u * v)
}

#[test]
fn projection_of_a_bilinear_field() {
    let mut h = bilinear_patch();
    build_projections(&mut h, false).unwrap();
    assert!(h.proj.is_empty(), "no projection without the scattered diffraction");
    build_projections(&mut h, true).unwrap();
    assert_eq!(h.proj.len(), 1);
    assert_eq!(projection_bytes(&h), 4 * (4 * 8 + 4));
    // a bilinear field is in the space: the coefficients are its corner values
    let vals: Vec<Complex64> = h.inner.iter().map(|p| bilinear(p.uf, p.vf)).collect();
    let mut coef = Vec::new();
    project(&h, 0, &vals, &mut coef).unwrap();
    assert_eq!(coef.len(), 4);
    let corners = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // u fastest
    for (k, &(u, v)) in corners.iter().enumerate() {
        let e = (coef[k] - bilinear(u, v)).norm();
        assert!(e < 1e-13, "coefficient {k}: {} vs {}, error {e:e}", coef[k], bilinear(u, v));
    }
}

#[test]
fn spline_values_of_a_bilinear_field() {
    let h = bilinear_patch();
    let corners = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)];
    let cv: Vec<Complex64> = corners.iter().map(|&(u, v)| bilinear(u, v)).collect();
    for &(u, v) in &[(0.3, -0.7), (-0.9, 0.2), (0.0, 0.0)] {
        let (mut f, mut fu, mut fv) = (Complex64::default(), Complex64::default(), Complex64::default());
        spline_values(&h, 0, &cv, u, v, 0, 0, &mut f, &mut fu, &mut fv);
        let du = Complex64::new(2.0 + 0.25 * v, v);
        let dv = Complex64::new(-0.5 + 0.25 * u, u);
        assert!((f - bilinear(u, v)).norm() < 1e-14);
        assert!((fu - du).norm() < 1e-14);
        assert!((fv - dv).norm() < 1e-14);
    }
}

#[test]
fn surface_gradient_of_tangents() {
    // tangents xu = (1, 0, 0), xv = (1, 1, 0): the gradient g in their plane with g.xu = fu, g.xv = fv
    let xu = v3(1.0, 0.0, 0.0);
    let xv = v3(1.0, 1.0, 0.0);
    let fu = Complex64::new(2.0, -1.0);
    let fv = Complex64::new(5.0, 0.5);
    let mut g = [Complex64::default(); 3];
    surface_gradient(&xu, &xv, fu, fv, &mut g);
    let gu = g[0] * xu[0] + g[1] * xu[1] + g[2] * xu[2];
    let gv = g[0] * xv[0] + g[1] * xv[1] + g[2] * xv[2];
    assert!((gu - fu).norm() < 1e-14 && (gv - fv).norm() < 1e-14 && g[2].norm() == 0.0);
}

#[test]
fn mode_signs_and_indices() {
    // one body, the group {e, sx, sy, sxsy} acting as the same local reflections
    let h = HoGeom {
        ind: vec![vec![
            Reflection { sx: 1.0, sy: 1.0 },
            Reflection { sx: -1.0, sy: 1.0 },
            Reflection { sx: 1.0, sy: -1.0 },
            Reflection { sx: -1.0, sy: -1.0 },
        ]],
        ..HoGeom::default()
    };
    let expect = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0],
    ];
    for g in 0..4 {
        for m in 0..6 {
            assert_eq!(mode_sign(&h, g, m), expect[g as usize][m as usize], "element {g}, mode {m}");
        }
    }
    let c = Column { coord: 3, body: 0, mode: 4, ..Column::default() };
    assert_eq!(mode_index(&h, &c), 3);
    // a pressure mode: the modes of the earlier surfaces first
    let mut h2 = HoGeom { P0: 6, ps: vec![Default::default(), Default::default()], ..HoGeom::default() };
    h2.ps[0].shapes = vec![PressureShape::Uniform; 2];
    h2.ps[1].coord0 = 8;
    h2.ps[1].shapes = vec![PressureShape::Uniform; 3];
    let c = Column { coord: 9, body: 0, mode: 0, ps: 1, ..Column::default() };
    assert_eq!(mode_index(&h2, &c), 6 + 2 + 1);
}
