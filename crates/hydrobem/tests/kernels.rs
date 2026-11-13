//! Unit tests of the kernel helpers: the Chebyshev bases of the per-frequency tiles, the direct
//! (untiled) finite-depth kernel at coincident points on the free surface, the panel images of the
//! symmetry group and the strip-mode series of the channel lattice sum next to its Ewald form.

use hydrobem_core::internals::geometry::Reflection;
use hydrobem_core::internals::green_channel::channel_lattice;
use hydrobem_core::internals::green_fin::{FreqMode, finite_kernel, make_finite_depth};
use hydrobem_core::internals::low_order::Geometry;
use hydrobem_core::internals::quadrature::{chebyshev_cosines, chebyshev_nodes, chebyshev_power_basis};

// The power basis reproduces T_j(t) = cos(j acos t), the nodes are the zeros of T_N and the cosine
// matrix is orthogonal on them (sum_i T_j T_k = N/2 for j = k > 0, N for j = k = 0, else 0): the
// identities the tiles of the finite-depth and the channel kernels rest on.
#[test]
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn chebyshev_bases() {
    const N: usize = 12;
    let a = chebyshev_power_basis::<N>();
    let t = chebyshev_nodes::<N>();
    let cs = chebyshev_cosines::<N>();
    for &x in &[-1.0, -0.73, 0.0, 0.41, 1.0] {
        for j in 0..N {
            let p: f64 = (0..N).map(|k| a[j][k] * f64::powi(x, k as i32)).sum();
            let exact = (j as f64 * f64::acos(x)).cos();
            assert!((p - exact).abs() < 1e-11, "T_{j}({x}) = {p} vs {exact}");
        }
    }
    for i in 0..N {
        assert!((N as f64 * t[i].acos()).cos().abs() < 1e-14, "node {i} is a zero of T_N");
    }
    for j in 0..N {
        for k in 0..N {
            let s: f64 = (0..N).map(|i| cs[j][i] * cs[k][i]).sum();
            let exact = if j != k {
                0.0
            } else if j == 0 {
                N as f64
            } else {
                N as f64 / 2.0
            };
            assert!((s - exact).abs() < 1e-12, "cosines {j} {k}: {s} vs {exact}");
        }
    }
}

// Coincident points on z = 0 (an interior panel at its own centroid): the direct evaluation of the
// integral form takes the limit of Ft + 2 psi as the tiled one does, so the two agree to the
// accuracy of the tiles (5e-7, the finite_tiles test of core.rs) instead of the direct form
// returning a non-finite value.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn direct_finite_kernel_at_coincident_points() {
    let h = 2.0;
    for H in [1e-3, 0.3, 2.0, 40.0] {
        let tiled = make_finite_depth(h, H / h, FreqMode::Ordinary, true);
        let direct = make_finite_depth(h, H / h, FreqMode::Ordinary, false);
        let a = finite_kernel(&tiled, 0.0, 0.0, 0.0, None, None);
        let b = finite_kernel(&direct, 0.0, 0.0, 0.0, None, None);
        let x = [b.Wr, b.WrR, b.WrZ, b.Wrz, b.Wi, b.WiR, b.WiZ, b.Wiz];
        let y = [a.Wr, a.WrR, a.WrZ, a.Wrz, a.Wi, a.WiR, a.WiZ, a.Wiz];
        for q in 0..8 {
            assert!(x[q].is_finite(), "H = {H}: entry {q} of the direct kernel finite");
            assert!((x[q] - y[q]).abs() <= 5e-7 * y[q].abs().max(1.0 / h), "H = {H}: entry {q}: {} vs {}", x[q], y[q]);
        }
    }
}

// The group {e, sx} of a stored panel and its x reflection: the panel g j of every panel j.
#[test]
fn panel_images_of_the_group() {
    let e = Reflection { sx: 1.0, sy: 1.0 };
    let rx = Reflection { sx: -1.0, sy: 1.0 };
    let g = Geometry { G: 2, Ns: 2, Nf: 4, refl: vec![e, rx], img: vec![0, 1, 2, 3], ..Geometry::default() };
    // e maps every panel onto itself, sx exchanges the stored panels and their images
    assert_eq!(g.panel_images(), vec![0, 1, 2, 3, 2, 3, 0, 1]);
}

// The lattice sum F(x, s) of the channel by the strip-mode series (x >= 2 a d) and by the Ewald
// form (below) agree where the two meet, and the series stops (it reaches its e^-40 term) also far
// from the lattice, where few modes are needed, and close to x = 2 a d, where many are.
#[test]
fn channel_lattice_forms_meet() {
    let (k, d) = (0.9, 6.0);
    let a = 0.4f64.min(2.0 / (k * d));
    let x0 = 2.0 * a * d;
    for s in [0.0, 1.3, 2.9] {
        let below = channel_lattice(x0 * (1.0 - 1e-9), s, k, d).unwrap();
        let above = channel_lattice(x0, s, k, d).unwrap();
        assert!((below - above).norm() <= 1e-8 * above.norm().max(1.0), "s = {s}: {below} vs {above}");
        assert!(channel_lattice(40.0 * d, s, k, d).unwrap().is_finite());
    }
}
