//! Unit tests of the group lo_b (lo_frequency, lo_velocity, tanks) and of the inline helpers of
//! low_order they use: the incident wave, the far-field kernel, the Kochin factor, the periodic
//! logarithm of the channel, the field values at points (and their independence of the thread
//! count) and the tank terms. The full solve of a frequency needs the other groups (statics,
//! motions) and is exercised by the regression tests once they are merged.

use hydrobem_core::internals::dispersion::dispersion_x0;
use hydrobem_core::internals::geometry::{Reflection, Vec3, v3};
use hydrobem_core::internals::green_channel::make_channel_tail;
use hydrobem_core::internals::green_fin::{FreqMode, fin_point, make_finite_depth};
use hydrobem_core::internals::lo_frequency::field_values;
use hydrobem_core::internals::low_order::{
    FreqCtx, Geometry, Output, Plan, Static, StaticEntry, SurfaceKind, TankField, channel_far, channel_far_sheet,
    far_kernel, incident, influence, kochin_global, periodic_log_regular,
};
use hydrobem_core::internals::panel_integrals::PanelGeom;
use hydrobem_core::internals::problem::Problem;
use hydrobem_core::internals::progress::SolveContext;
use hydrobem_core::internals::tanks::{add_tank_terms, field_all};
use num_complex::{Complex32, Complex64};

const G: f64 = 9.80665;
const CZ: Complex64 = Complex64::new(0.0, 0.0);

/// A small deterministic generator (uniform in [-1, 1)).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn plan(heads: &[f64]) -> Plan {
    Plan { grav: G, rho: 1025.0, heads: heads.to_vec(), diffraction: true, ..Plan::default() }
}

fn ctx(w: f64, depth: f64, k: f64) -> FreqCtx {
    FreqCtx { mode: FreqMode::Ordinary, w, K: w * w / G, k, depth, logc: 1.0, ..FreqCtx::default() }
}

fn geom(walls: Vec<Reflection>) -> Geometry {
    Geometry { walls, yaw: 30.0, px: 1.5, py: -2.0, ..Geometry::default() }
}

fn ident() -> Reflection {
    Reflection { sx: 1.0, sy: 1.0 }
}

fn close(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() <= tol * (1.0 + b.norm())
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn incident_gradient_and_walls() {
    // the gradient of the incident wave is that of its value (central differences), in infinite
    // and finite depth; with walls the wave is the sum over the reflected points
    let w = 1.1;
    let K = w * w / G;
    let h = 12.0;
    // finite depth: k tanh kh = K
    let mut kf = K;
    for _ in 0..100 {
        kf = K / (kf * h).tanh();
    }
    for (depth, k) in [(f64::INFINITY, K), (h, kf)] {
        let fc = ctx(w, depth, k);
        let pl = plan(&[0.3]);
        let walls = [vec![ident()], vec![ident(), Reflection { sx: 1.0, sy: -1.0 }]];
        for wl in walls {
            let g = geom(wl.clone());
            let x = v3(0.7, -1.2, -2.5);
            let (p, gr) = incident(&g, &pl, &fc, 0.3, &x);
            let d = 1e-5;
            for i in 0..3 {
                let (mut xp, mut xm) = (x, x);
                xp[i] += d;
                xm[i] -= d;
                let (pp, _) = incident(&g, &pl, &fc, 0.3, &xp);
                let (pm, _) = incident(&g, &pl, &fc, 0.3, &xm);
                let fd = (pp - pm) / (2.0 * d);
                assert!(close(gr[i], fd, 1e-7), "incident gradient {i}: {:?} vs {:?}", gr[i], fd);
            }
            // unit amplitude: |phi| = g / w Z(z)
            if wl.len() == 1 {
                let Z = if depth.is_finite() { (k * (x[2] + h)).cosh() / (k * h).cosh() } else { (k * x[2]).exp() };
                assert!((p.norm() - G / w * Z).abs() < 1e-12 * G / w);
            }
        }
    }
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn far_kernel_derivatives() {
    // gn = n . grad g and gt, gnt the theta derivatives (central differences)
    let w = 0.9;
    let K = w * w / G;
    for depth in [f64::INFINITY, 30.0] {
        let fc = ctx(w, depth, K * 1.05);
        let x = v3(2.0, -1.0, -3.0);
        let nn = 1.0f64 / 3.0f64.sqrt();
        let n = v3(nn, -nn, nn);
        let th: f64 = 0.4;
        let f = far_kernel(&fc, &x, &n, th.cos(), th.sin());
        let d = 1e-6;
        let mut gn = CZ;
        for i in 0..3 {
            let (mut xp, mut xm) = (x, x);
            xp[i] += d;
            xm[i] -= d;
            let fp = far_kernel(&fc, &xp, &n, th.cos(), th.sin());
            let fm = far_kernel(&fc, &xm, &n, th.cos(), th.sin());
            gn += (fp.g - fm.g) / (2.0 * d) * n[i];
        }
        assert!(close(f.gn, gn, 1e-7));
        let fp = far_kernel(&fc, &x, &n, (th + d).cos(), (th + d).sin());
        let fm = far_kernel(&fc, &x, &n, (th - d).cos(), (th - d).sin());
        assert!(close(f.gt, (fp.g - fm.g) / (2.0 * d), 1e-7));
        assert!(close(f.gnt, (fp.gn - fm.gn) / (2.0 * d), 1e-7));
    }
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn kochin_global_factor() {
    // H exp(i k (px cos + py sin)) and its theta derivative
    let fc = ctx(1.0, f64::INFINITY, 0.25);
    let (px, py, th) = (3.0, -2.0, 0.7);
    let H0 = Complex64::new(0.3, -1.1);
    let Ht0 = Complex64::new(-0.2, 0.5);
    let e = |t: f64| {
        let ph = fc.k * (px * t.cos() + py * t.sin());
        Complex64::new(ph.cos(), ph.sin())
    };
    let (H, Ht) = kochin_global(&fc, px, py, th, H0, Ht0);
    assert!(close(H, e(th) * H0, 1e-14));
    // d/dtheta (e H) with H linear in theta of slope Ht0
    let d = 1e-6;
    let fd = (e(th + d) * (H0 + d * Ht0) - e(th - d) * (H0 - d * Ht0)) / (2.0 * d);
    assert!(close(Ht, fd, 1e-8));
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn periodic_log_derivatives() {
    // va, vs are the derivatives of v, and the series near z = 0 joins the closed form
    let W = 3.0;
    for (a, s) in [(0.4, 0.3), (-1.7, 2.2), (5.0, -0.8), (1e-4, 2e-4)] {
        let (v, va, vs) = periodic_log_regular(a, s, W);
        let d = 1e-6;
        let f = |a: f64, s: f64| {
            let (v, _, _) = periodic_log_regular(a, s, W);
            v
        };
        assert!((va - (f(a + d, s) - f(a - d, s)) / (2.0 * d)).abs() < 1e-6, "va at ({a}, {s})");
        assert!((vs - (f(a, s + d) - f(a, s - d)) / (2.0 * d)).abs() < 1e-6, "vs at ({a}, {s})");
        // log|2 sinh(f z) / z| directly
        let fz = std::f64::consts::PI / (2.0 * W) * Complex64::new(a, s);
        let ex = (2.0 * fz.sinh()).norm().ln() - Complex64::new(a, s).norm().ln();
        assert!((v - ex).abs() < 1e-10, "v at ({a}, {s}): {v} vs {ex}");
    }
}

/// A geometry of Nf wetted panels (centroid, normal and area only: the point influences use
/// the kernel at the centroid and the static entries) and its static rows at P points.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn field_case(Nf: usize, P: usize, rng: &mut Lcg) -> (Geometry, Static<f32>, Vec<Vec3>) {
    let mut g = geom(vec![ident()]);
    g.Nf = Nf as i32;
    for _ in 0..Nf {
        let mut p = PanelGeom {
            c: v3(3.0 * rng.next(), 3.0 * rng.next(), -1.0 - rng.next().abs() * 4.0),
            ..PanelGeom::default()
        };
        let n = v3(rng.next(), rng.next(), rng.next());
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        p.n = v3(n[0] / l, n[1] / l, n[2] / l);
        p.area = 0.2 + rng.next().abs();
        g.pg.push(p);
        g.kind.push(SurfaceKind::Wet);
        g.body.push(0);
        g.mirror.push(0);
    }
    let mut st = Static::<f32>::default();
    for _ in 0..P * Nf {
        st.e.push(StaticEntry {
            rs: rng.next() as f32,
            rd: rng.next() as f32,
            dl: 0.01 * rng.next() as f32,
            dd: 0.01 * rng.next() as f32,
        });
    }
    let pts = (0..P).map(|_| v3(6.0 * rng.next(), 6.0 * rng.next(), -0.5 - 3.0 * rng.next().abs())).collect();
    (g, st, pts)
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn field_values_threads_and_values() {
    // the field outputs do not depend on the thread count and are Green's identity of the
    // point influences: frad = w^2/g sum (phi D - S u) / 4 pi, fsca = -i w/g (phi_S + phi_I) . D / 4 pi
    let mut rng = Lcg(12345);
    let (Nf, P, nrad, nh) = (7usize, 9usize, 2usize, 2usize);
    let (g, st, pts) = field_case(Nf, P, &mut rng);
    let w = 0.8;
    let fc = ctx(w, f64::INFINITY, w * w / G);
    let pl = plan(&[0.0, 1.2]);
    let rnd =
        |rng: &mut Lcg, n: usize| (0..n).map(|_| Complex64::new(rng.next(), rng.next())).collect::<Vec<Complex64>>();
    let phi = rnd(&mut rng, (nrad + nh) * Nf);
    let u = rnd(&mut rng, nrad * Nf);
    let phiI = rnd(&mut rng, nh * Nf);
    let run = |threads: i32| {
        let mut frad = vec![Complex32::new(0.0, 0.0); P * nrad];
        let mut finc = vec![Complex32::new(0.0, 0.0); P * nh];
        let mut fsca = finc.clone();
        let mut ftot = finc.clone();
        let pool = hydrobem_core::internals::parallel::pool("test", threads).unwrap();
        pool.install(|| {
            field_values::<f32, Complex64>(
                &g,
                &pl,
                &st,
                &fc,
                &pts,
                nrad as i32,
                nh as i32,
                &phi,
                &u,
                &phiI,
                &[],
                &mut frad,
                &mut finc,
                &mut fsca,
                &mut ftot,
                SolveContext::default().statics(),
                None,
            )
        })
        .unwrap();
        (frad, finc, fsca, ftot)
    };
    let one = run(1);
    for threads in [2, 4, 16] {
        assert_eq!(one, run(threads), "field values with {threads} threads");
    }
    let (frad, finc, fsca, ftot) = one;
    let f32c = |z: Complex64| Complex32::new(z.re as f32, z.im as f32);
    for q in 0..P {
        let (mut S, mut D) = (vec![CZ; Nf], vec![CZ; Nf]);
        for j in 0..Nf {
            (S[j], D[j]) = influence(&g, &st, q * Nf, j as i32, &pts[q], &fc);
        }
        for c in 0..nrad {
            let mut v = CZ;
            for j in 0..Nf {
                v += phi[c * Nf + j] * D[j] - S[j] * u[c * Nf + j];
            }
            assert_eq!(frad[q * nrad + c], f32c(w * w / G * v / (4.0 * std::f64::consts::PI)));
        }
        for h in 0..nh {
            let mut v = CZ;
            for j in 0..Nf {
                v += phi[(nrad + h) * Nf + j] * D[j];
                v += phiI[h * Nf + j] * D[j];
            }
            let etaS = v / (4.0 * std::f64::consts::PI) * Complex64::new(0.0, -w / G);
            let (pI, _) = incident(&g, &pl, &fc, pl.heads[h], &pts[q]);
            let etaI = pI * Complex64::new(0.0, -w / G);
            assert_eq!(fsca[q * nh + h], f32c(etaS));
            assert_eq!(finc[q * nh + h], f32c(etaI));
            assert_eq!(ftot[q * nh + h], f32c(etaI + etaS));
        }
    }
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn tank_terms_and_field_all() {
    // the tank added mass and damping join the body's at an ordinary frequency, the limit added
    // masses at omega = 0 and inf
    let (F, D, K) = (2usize, 3usize, 2usize);
    let mut pl = Plan { D: D as i32, ..Plan::default() };
    pl.rad = vec![Default::default(); K];
    pl.tA = (0..F * D * K).map(|i| i as f64).collect();
    pl.tB = (0..F * D * K).map(|i| 0.5 * i as f64).collect();
    pl.tA0 = (0..D * K).map(|i| 10.0 + i as f64).collect();
    pl.tAinf = (0..D * K).map(|i| 20.0 + i as f64).collect();
    // the outputs of one frequency, here the second
    let mut out = Output {
        A: vec![1.0; D * K],
        B: vec![2.0; D * K],
        A0: vec![0.0; D * K],
        Ainf: vec![0.0; D * K],
        ..Output::default()
    };
    add_tank_terms(&pl, FreqMode::Ordinary, 1, &mut out);
    for k in 0..D * K {
        assert_eq!(out.A[k], 1.0 + (D * K + k) as f32);
        assert_eq!(out.B[k], 2.0 + 0.5 * (D * K + k) as f32);
    }
    add_tank_terms(&pl, FreqMode::Zero, -1, &mut out);
    add_tank_terms(&pl, FreqMode::Infinite, -1, &mut out);
    for k in 0..D * K {
        assert_eq!(out.A0[k], 10.0 + k as f32);
        assert_eq!(out.Ainf[k], 20.0 + k as f32);
    }
    // no tanks: nothing added
    let mut out2 = out.clone();
    add_tank_terms(&Plan::default(), FreqMode::Ordinary, 0, &mut out2);
    assert_eq!(out2.A, out.A);

    // field_all: the exterior points from ext, the tank points from their tank problems
    let mut m = Problem::default();
    m.field_points = vec![v3(0.0, 0.0, -1.0); 3];
    m.field_tanks = vec![0, 1, 0];
    let n = K;
    let ext: Vec<Complex32> = (0..F * 2 * n).map(|i| Complex32::new(i as f32, 0.0)).collect();
    let tf = TankField { point: 1, rad: (0..F * K).map(|i| Complex32::new(0.0, i as f32)).collect(), vel: Vec::new() };
    pl.tfield = vec![tf];
    let all = field_all(&m, &pl, ext.clone(), F as i32, n as i32, true, false);
    for f in 0..F {
        for e in 0..n {
            assert_eq!(all[(f * 3) * n + e], ext[(f * 2) * n + e]);
            assert_eq!(all[(f * 3 + 1) * n + e], Complex32::new(0.0, (f * K + e) as f32));
            assert_eq!(all[(f * 3 + 2) * n + e], ext[(f * 2 + 1) * n + e]);
        }
    }
    let dif = field_all(&m, &pl, ext.clone(), F as i32, n as i32, false, false);
    assert!(dif[n..2 * n].iter().all(|z| *z == Complex32::new(0.0, 0.0)));
    // without tank points ext itself
    m.field_tanks.clear();
    assert_eq!(field_all(&m, &pl, ext.clone(), F as i32, n as i32, true, false), ext);
}

#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn channel_far_sheet_is_the_derivative_of_channel_far() {
    // the far images of the sheet rows: Kp = nx . grad_x of S_far and H = nx . grad_x of D_far by
    // central differences of channel_far (the field point moved along nx), at an ordinary
    // frequency (evanescent series and tail polynomials), at omega = 0 (the periodic log) and
    // omega = inf (the series alone); the source point anywhere in the channel
    let (h, W) = (10.0, 8.0);
    let mut rnd = Lcg(7);
    for mode in [FreqMode::Ordinary, FreqMode::Zero, FreqMode::Infinite] {
        let mut fc = FreqCtx { mode, depth: h, ..FreqCtx::default() };
        if mode == FreqMode::Ordinary {
            fc.w = 1.2;
            fc.K = fc.w * fc.w / G;
            fc.k = dispersion_x0(fc.K * h) / h;
        }
        fc.fd = make_finite_depth(h, fc.K, mode, true);
        fc.Me = 1 + (18.0 / (fc.fd.kn[0] * W)).ceil() as i32;
        if mode == FreqMode::Ordinary {
            fc.ch = make_channel_tail(W, fc.fd.k0, 30.0).unwrap();
        }
        let (mut err, mut size) = (0.0f64, 0.0f64);
        for _ in 0..40 {
            let pt = |r: &mut Lcg| v3(10.0 * r.next(), 0.4 * W * r.next(), -h * (0.5 + 0.4 * r.next()));
            let dir = |r: &mut Lcg| {
                let v = v3(r.next(), r.next(), r.next());
                (1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()) * v
            };
            let (x, nx, xi, n) = (pt(&mut rnd), dir(&mut rnd), pt(&mut rnd), dir(&mut rnd));
            let pj = fin_point(&fc.fd, xi[2]);
            let far = |y: &Vec3| channel_far(&fc, W, y, &xi, &n, 1.0, &fin_point(&fc.fd, y[2]), &pj);
            let d = 1e-4;
            let (sp, dp) = far(&(x + d * nx));
            let (sm, dm) = far(&(x + (-d) * nx));
            let (H, Kp) = channel_far_sheet(&fc, W, &x, &nx, &xi, &n, 1.0, &fin_point(&fc.fd, x[2]), &pj);
            let (Hd, Kd) = ((dp - dm) / (2.0 * d), (sp - sm) / (2.0 * d));
            err = err.max((H - Hd).norm()).max((Kp - Kd).norm());
            size = size.max(Hd.norm()).max(Kd.norm());
        }
        println!("channel_far_sheet {mode:?}: {:.2e} of the largest value {size:.2e}", err / size);
        assert!(err < 1e-8 * size, "{mode:?}: {:.2e}", err / size);
    }
}
