//! The momentum drift of a solve and the mode displacements at points.
//!
//! The mean drift by momentum conservation from the Kochin functions and the motions of a solve
//! (theory §11.1), and the displacements of the modes at points (theory §3.1).

use crate::codes::QuantityId;
use crate::mesh_desc::PatchKind;
use num_complex::{Complex32, Complex64};

use crate::constants::PI;
use crate::dispersion::dispersion_x0;
use crate::error::{Error, Result, at_body_element};
use crate::geometry::{Vec3, cross, v3};
use crate::mesh::patch_eval;
use crate::problem::Problem;
use crate::results::{Element, Results, elements_of, put};

// the elements of quantity q (empty when absent)
fn get<T: Element>(r: &Results, q: QuantityId) -> Vec<T> {
    r.quantities.get(&q).map(|it| elements_of(it.data())).unwrap_or_default()
}

fn has(r: &Results, q: QuantityId) -> bool {
    r.quantities.contains_key(&q)
}

fn dims(r: &Results, q: QuantityId) -> &[i64] {
    &r.quantities[&q].dims
}

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn wave_number(w: f64, depth: f64, g: f64) -> f64 {
    let K = w * w / g;
    if depth.is_finite() { dispersion_x0(K * depth) / depth } else { K }
}

fn ci(im: f64) -> Complex64 {
    Complex64::new(0.0, im)
}

fn of_f(v: Complex32) -> Complex64 {
    Complex64::new(v.re as f64, v.im as f64)
}

// The trigonometric interpolant of N samples v at 2 pi t / N: the Fourier coefficients
// c_m = (1/N) sum_t v_t e^{-i m 2 pi t / N}, |m| < N/2, and for an even N the Nyquist term split
// equally between m = +-N/2 (a real cos(N beta / 2) term). The coefficients of a series are formed
// once from the table of the N-th roots of unity and evaluated at every heading with the phases of
// the heading, O(N^2) per series and O(N) per value.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct Trig {
    N: i32,
    lo: i32,               // the modes m = lo .. -lo (lo = -(N - 1) / 2)
    roots: Vec<Complex64>, // e^{-2 pi i k / N}, k = 0 .. N - 1
}

impl Trig {
    fn new(n: i32) -> Trig {
        let mut roots = vec![Complex64::new(0.0, 0.0); n as usize];
        for k in 0..n {
            roots[k as usize] = ci(-2.0 * PI * k as f64 / n as f64).exp();
        }
        Trig { N: n, lo: -(n - 1) / 2, roots }
    }
    // the coefficients c_lo .. c_-lo, then (even N) the Nyquist coefficient, each divided by N
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn coefficients(&self, v: &[Complex64]) -> Vec<Complex64> {
        let N = self.N;
        let mut c = Vec::new();
        for m in self.lo..=-self.lo {
            let mut sum = Complex64::new(0.0, 0.0);
            let step = ((m % N) + N) % N;
            let mut k = 0;
            for t in 0..N {
                sum += v[t as usize] * self.roots[k as usize];
                k = (k + step) % N;
            }
            c.push(sum / N as f64);
        }
        if N % 2 == 0 {
            let mut sum = Complex64::new(0.0, 0.0);
            for t in 0..N {
                sum += (if t % 2 != 0 { -1.0 } else { 1.0 }) * v[t as usize];
            }
            c.push(sum / N as f64);
        }
        c
    }
    // the phases of beta: e^{i m beta} of every mode, then cos(N beta / 2)
    fn phases(&self, beta: f64) -> Vec<Complex64> {
        let mut e = Vec::new();
        for m in self.lo..=-self.lo {
            e.push(ci(m as f64 * beta).exp());
        }
        if self.N % 2 == 0 {
            e.push(Complex64::new(((self.N / 2) as f64 * beta).cos(), 0.0));
        }
        e
    }
    fn value(c: &[Complex64], e: &[Complex64]) -> Complex64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for k in 0..c.len() {
            sum += c[k] * e[k];
        }
        sum
    }
}

/// The Kochin angle count of second_order.drift_momentum_conservation_angles = "auto" (-1 in the problem): 8 ceil((2 ceil(k R) + 96) / 8)
/// for the largest wave number k and the horizontal radius R of the bodies about the origin.
pub fn auto_kochin_angles(m: &Problem) -> i32 {
    // the Kochin function of a body within the horizontal radius R of the origin has Fourier
    // modes |m| <~ kR; the trapezoidal rule of N points integrates |H|^2 (modes up to 2 kR + O(1))
    // exactly for N > 2 kR plus the margin
    let mut k: f64 = 0.0;
    let mut any = false;
    for &w in &m.omegas {
        if w > 0.0 && w.is_finite() {
            k = k.max(wave_number(w, m.env.depth, m.env.gravity));
            any = true;
        }
    }
    if !any {
        return 0;
    }
    // the horizontal radius of a mesh about its body's origin
    let extent = |mesh: &crate::mesh::Mesh| {
        let mut local: f64 = 0.0;
        let mut take = |x: Vec3| local = local.max(x[0].hypot(x[1]));
        if !mesh.patch {
            for p in &mesh.panels {
                for x in &p.raw {
                    take(*x);
                }
            }
        } else if mesh.geometry == PatchKind::Flat && !mesh.trimmed() {
            let mut i = 0;
            while i + 2 < mesh.vertices.len() {
                take(v3(mesh.vertices[i], mesh.vertices[i + 1], 0.0));
                i += 3;
            }
        } else if mesh.geometry == PatchKind::Spline && !mesh.trimmed() {
            let mut i = 0;
            while i + 2 < mesh.coefficients.len() {
                take(v3(mesh.coefficients[i], mesh.coefficients[i + 1], 0.0));
                i += 3;
            }
        } else {
            // analytic or trimmed: the patch maps on a 9 x 9 grid
            for p in 0..mesh.count {
                for i in 0..9 {
                    for j in 0..9 {
                        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                        patch_eval(mesh, p, -1.0 + 0.25 * j as f64, -1.0 + 0.25 * i as f64, &mut x, &mut xu, &mut xv);
                        take(x);
                    }
                }
            }
        }
        local
    };
    let mut radius: f64 = 0.0;
    for b in &m.bodies {
        // the body mesh and the free-surface plates of meshes of their own (they radiate too)
        let mut local = extent(&m.meshes[b.mesh as usize]);
        for p in &b.plates {
            if let Some(pi) = p.mesh {
                local = local.max(extent(&m.meshes[pi]));
            }
        }
        radius = radius.max(b.position[0].hypot(b.position[1]) + local);
    }
    // the bandwidth kR + O((kR)^(1/3)) of the Kochin functions: the measured need is
    // 2 kR + 10 (kR)^(1/3), so the margin 96 holds up to kR ~ 900
    let n = 2 * (k * radius).ceil() as i32 + 96;
    8 * ((n + 7) / 8)
}

/// QuantityId::DriftMomentumConservation [F, P, 3] of the drift heading pairs (and
/// QuantityId::DriftHeadingPairs) from the Kochin functions and motions of r (theory §11.1); nothing
/// without the Kochin functions.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn add_momentum_drift(m: &Problem, r: &mut Results) -> Result<()> {
    if !has(r, QuantityId::KochinDiffraction) {
        return Ok(());
    }
    let (rho, g, depth) = (m.env.density, m.env.gravity, m.env.depth);
    let theta: Vec<f64> = get(r, QuantityId::KochinAngles);
    let omegas: Vec<f64> = get(r, QuantityId::Omegas);
    let heads: Vec<f64> = get(r, QuantityId::Headings);
    let hd: Vec<Complex32> = get(r, QuantityId::KochinDiffraction);
    let hdt: Vec<Complex32> = get(r, QuantityId::KochinDiffractionDerivative);
    let hr: Vec<Complex32> = get(r, QuantityId::KochinRadiation);
    let hrt: Vec<Complex32> = get(r, QuantityId::KochinRadiationDerivative);
    let xi: Vec<Complex32> = get(r, QuantityId::Motion);
    let ids: Vec<i32> = get(r, QuantityId::RadiationModeIds);
    let (F, H, N) = (omegas.len(), heads.len(), theta.len());
    let K = ids.len();
    let D = if has(r, QuantityId::Motion) { dims(r, QuantityId::Motion)[2] as usize } else { 0 };
    let moving = !hr.is_empty() && !xi.is_empty() && K > 0;
    let dth = 2.0 * PI / N as f64;
    let mut pairs = vec![Complex64::new(0.0, 0.0); F * H * H * 3];
    let mut uni = vec![0.0f64; F * H * 3];
    let mut beta = vec![0.0f64; H];
    for h in 0..H {
        beta[h] = heads[h] * PI / 180.0;
    }
    let trig = Trig::new(N as i32);
    let mut phase: Vec<Vec<Complex64>> = vec![Vec::new(); H];
    for h in 0..H {
        phase[h] = trig.phases(beta[h]);
    }
    for f in 0..F {
        if m.cancel.get() {
            return Err(Error::cancelled("the solve was cancelled")); // between frequencies
        }
        let w = omegas[f];
        let K0 = w * w / g;
        let k = wave_number(w, depth, g);
        // C (the propagating part of G is -i C Z(z) Z(zeta) H0(kR)) and K_e = 2 k^2 int Z^2 dz
        let mut C = 2.0 * PI * K0;
        let mut Ke = K0;
        if depth.is_finite() {
            let x = k * depth;
            let A0 = 2.0 * PI * k / (-(-4.0 * x).exp_m1() + 4.0 * x * (-2.0 * x).exp());
            C = A0 * (1.0 + (-2.0 * x).exp()).powi(2);
            Ke = K0 + k * k * depth / x.cosh().powi(2);
        }
        let fac = -(ci(C) / (4.0 * PI)) * (2.0 / (PI * k)).sqrt() * ci(PI / 4.0).exp();
        let a = ci(g / w);
        let e = ci(-PI / 4.0).exp();
        let q = (2.0 * PI / k).sqrt();
        let mut b = vec![vec![Complex64::new(0.0, 0.0); N]; H];
        let mut bt = b.clone();
        for h in 0..H {
            for t in 0..N {
                let o = (f * N + t) * H + h;
                let mut v = of_f(hd[o]);
                let mut vt = of_f(hdt[o]);
                if moving {
                    for c in 0..K {
                        let x = of_f(xi[(f * H + h) * D + ids[c] as usize - 1]);
                        let oc = (f * N + t) * K + c;
                        v += ci(w) * of_f(hr[oc]) * x;
                        vt += ci(w) * of_f(hrt[oc]) * x;
                    }
                }
                b[h][t] = fac * v;
                bt[h][t] = fac * vt;
            }
        }
        let mut bb = vec![Complex64::new(0.0, 0.0); H * H];
        let mut bbt = vec![Complex64::new(0.0, 0.0); H * H];
        for i in 0..H {
            let c = trig.coefficients(&b[i]);
            let ct = trig.coefficients(&bt[i]);
            for j in 0..H {
                bb[i * H + j] = Trig::value(&c, &phase[j]);
                bbt[i * H + j] = Trig::value(&ct, &phase[j]);
            }
        }
        let s = rho * Ke / 4.0;
        for i in 0..H {
            for j in 0..H {
                let (mut sx, mut sy, mut dd) =
                    (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
                for t in 0..N {
                    let pij = b[i][t] * b[j][t].conj();
                    sx += pij * theta[t].cos();
                    sy += pij * theta[t].sin();
                    dd += bt[i][t] * b[j][t].conj() - b[i][t] * bt[j][t].conj();
                }
                dd *= ci(-0.5); // / (2 i)
                let cx = -s * q * a.conj() * e * bb[i * H + j]; // incident j, disturbance i
                let cy = -s * q * a * e.conj() * bb[j * H + i].conj(); // incident i, disturbance j
                let o = &mut pairs[((f * H + i) * H + j) * 3..((f * H + i) * H + j) * 3 + 3];
                o[0] = -s * sx * dth + cx * beta[j].cos() + cy * beta[i].cos();
                o[1] = -s * sy * dth + cx * beta[j].sin() + cy * beta[i].sin();
                o[2] = (s / k) * dd * dth
                    - (s / k)
                        * q
                        * (ci(1.0) * a.conj() * e * bbt[i * H + j] - ci(1.0) * a * e.conj() * bbt[j * H + i].conj());
                if i == j {
                    for c in 0..3 {
                        uni[(f * H + i) * 3 + c] = o[c].re;
                    }
                }
            }
        }
    }
    // the heading pairs asked for: the equal headings as the real coefficients above
    let hp = crate::heading_pairs::heading_pairs(H, m.opt.drift_pairs);
    let mut v = Vec::with_capacity(F * hp.len() * 3);
    for f in 0..F {
        for &(i, j) in &hp {
            for c in 0..3 {
                v.push(if i == j {
                    Complex64::new(uni[(f * H + i) * 3 + c], 0.0)
                } else {
                    pairs[((f * H + i) * H + j) * 3 + c]
                });
            }
        }
    }
    put(r, QuantityId::DriftMomentumConservation, vec![F as i64, hp.len() as i64, 3], &v);
    crate::heading_pairs::put_pairs(r, QuantityId::DriftHeadingPairs, &heads, &hp);
    Ok(())
}

/// The displacement [n][D][3] (global axes) of every coordinate at the global points, each on the
/// one-based stored element of its body's mesh: the rigid modes about R and the generalized modes.
/// bodies, elements: [n]; points: [n][3]; out: [n][D][3].
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn mode_displacements(
    m: &Problem,
    n: i32,
    bodies: &[i32],
    records: &[i32],
    points: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let D = m.coordinates() as usize;
    let n = n as usize;
    out[..n * D * 3].fill(0.0);
    for i in 0..n {
        let bi = bodies[i];
        let rec = records[i];
        let here = at_body_element(bi, rec);
        if bi < 0 || bi >= m.bodies.len() as i32 {
            return Err(Error::value(format!("point {}: body {} does not exist", i + 1, bi)).at(here));
        }
        let b = &m.bodies[bi as usize];
        // the stored element of the body's mesh as an element of the mesh the body uses (the elements
        // cut from one stored element are in the same pieces of the generalized modes)
        let count = m.stored_records(b.mesh);
        if rec < 1 || rec > count {
            return Err(Error::value(format!("point {}: element {} does not exist", i + 1, rec)).at(here));
        }
        let Some(&used) = m.records_of_stored(b.mesh, rec).first() else {
            return Err(Error::value(format!(
                "point {}: element {} of the body's mesh is not used by the body (trimmed away, or of no surface)",
                i + 1,
                rec
            ))
            .at(here));
        };
        let rec = used + 1;
        // body axes: x_global = Rz(yaw) x_local + position
        let c = (b.yaw * PI / 180.0).cos();
        let s = (b.yaw * PI / 180.0).sin();
        let p = &points[3 * i..3 * i + 3];
        let dx = p[0] - b.position[0];
        let dy = p[1] - b.position[1];
        let xl = v3(c * dx + s * dy, -s * dx + c * dy, p[2] - b.position[2]);
        let mut store = |coord: usize, u: Vec3| {
            let o = &mut out[(i * D + coord) * 3..(i * D + coord) * 3 + 3];
            o[0] = c * u[0] - s * u[1];
            o[1] = s * u[0] + c * u[1];
            o[2] = u[2];
        };
        let off = m.derived()[bi as usize].coordinate_offset as usize;
        let r = xl - b.rotation_center;
        // unit rigid modes about R in body axes: translations e_j, rotations e_j x (x - R)
        for j in 0..3 {
            let mut e = v3(0.0, 0.0, 0.0);
            e[j] = 1.0;
            store(off + j, e);
            store(off + 3 + j, cross(e, r));
        }
        for k in 0..b.generalized.len() {
            store(off + 6 + k, b.generalized[k].displacement(&xl, rec));
        }
    }
    Ok(())
}
