//! The bodies in the potential QTF: what the sub-problems hold of them (the body points, the
//! coordinates and the radiation columns), the waterline values of the free surface's line
//! integral (theory §12.8) and the body condition of the first-order motions (§12.7, §12.15).

use num_complex::Complex64;

use super::zones::{BodySide, WlSeg};
use super::{CZ, I, read_c32, read_f64, read_i32};
use crate::codes::QuantityId;
use crate::constants::PI;
use crate::error::{Error, Result};
use crate::geometry::{Vec3, v3};
use crate::problem::Problem;
use crate::results::Results;

pub(super) type C3 = [Complex64; 3];

fn dot(a: &C3, b: &C3) -> Complex64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dotr(a: &C3, n: &[f64; 3]) -> Complex64 {
    a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
}
fn cross(a: &C3, b: &C3) -> C3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn crossr(a: &C3, n: &[f64; 3]) -> C3 {
    [a[1] * n[2] - a[2] * n[1], a[2] * n[0] - a[0] * n[2], a[0] * n[1] - a[1] * n[0]]
}
fn conj3(a: &C3) -> C3 {
    [a[0].conj(), a[1].conj(), a[2].conj()]
}

/// A body's yaw as the rotation from global to body axes (horizontal components).
fn to_body(b: &crate::problem::Body, v: [f64; 3]) -> [f64; 3] {
    let (cy, sy) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
    [cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2]]
}

/// The rigid generalized normals (n, d × n) of a point at d from the rotation centre with the
/// normal n (body axes).
fn rigid_normals(d: &[f64; 3], n: &[f64; 3]) -> [f64; 6] {
    [n[0], n[1], n[2], d[1] * n[2] - d[2] * n[1], d[2] * n[0] - d[0] * n[2], d[0] * n[1] - d[1] * n[0]]
}

/// What a sub-problem's results hold of the bodies: the body points (global) with their normals
/// (into the fluid) and areas, the body of every point, the first coordinate of each body (its six
/// rigid modes, then its generalized modes: the columns of the loads and of the radiation at Ω),
/// the coordinates nl, the free coordinates of each body with their radiation columns in the
/// first-order problem (nc of them), the element of every body point and whether it is an element
/// of a free-surface plate.
pub(super) struct Layout {
    pub(super) pts: Vec<f64>,
    pub(super) nrm: Vec<f64>,
    pub(super) area: Vec<f64>,
    pub(super) owner: Vec<usize>,
    pub(super) off: Vec<usize>,
    pub(super) nl: usize,
    pub(super) free: Vec<Vec<(usize, usize)>>,
    pub(super) nc: usize,
    pub(super) rec: Vec<i32>,
    pub(super) plate: Vec<bool>,
}

impl Layout {
    /// The layout of the body points of the first-order results ra (the body panels: the
    /// sub-problems have no body points of their own) for the coordinates of m; `moving`: the
    /// radiation of the free coordinates is in ra.
    pub(super) fn new(m: &Problem, ra: &Results, moving: bool) -> Result<Layout> {
        let off: Vec<usize> =
            m.bodies.iter().scan(0, |s, b| Some(std::mem::replace(s, *s + 6 + b.generalized.len()))).collect();
        let nl: usize = m.bodies.iter().map(|b| 6 + b.generalized.len()).sum();
        let mut nc = 0usize; // the radiation columns of all bodies
        let mut free: Vec<Vec<(usize, usize)>> = Vec::with_capacity(m.bodies.len());
        for (bi, b) in m.bodies.iter().enumerate() {
            let mut fr: Vec<(usize, usize)> = b.modes.iter().map(|&md| off[bi] + (md - 1) as usize).zip(nc..).collect();
            fr.extend((0..b.generalized.len()).map(|k| (off[bi] + 6 + k, nc + b.modes.len() + k)));
            nc += b.modes.len() + b.generalized.len();
            free.push(if moving { fr } else { Vec::new() });
        }
        if !moving {
            nc = 0;
        }
        let ids = read_i32(ra, QuantityId::BodyPointsIds);
        let owner: Vec<usize> = ids.chunks(4).map(|c| c[0] as usize).collect();
        let rec: Vec<i32> = ids.chunks(4).map(|c| c[1]).collect();
        let (pts, nrm, area) = (
            read_f64(ra, QuantityId::BodyPoints),
            read_f64(ra, QuantityId::BodyPointsNormals),
            read_f64(ra, QuantityId::BodyPointsAreas),
        );
        if owner.len() != area.len() || pts.len() != 3 * area.len() || nrm.len() != 3 * area.len() {
            return Err(Error::internal("potential QTF: unexpected sizes of the body points"));
        }
        let plate = owner.iter().zip(&rec).map(|(&b, &e)| crate::plate_geometry::is_plate_element(m, b, e)).collect();
        Ok(Layout { pts, nrm, area, owner, off, nl, free, nc, rec, plate })
    }
    /// The body points.
    pub(super) fn len(&self) -> usize {
        self.area.len()
    }
    /// The coordinates of body bi.
    pub(super) fn width(&self, m: &Problem, bi: usize) -> usize {
        6 + m.bodies[bi].generalized.len()
    }
    fn point(&self, q: usize) -> [f64; 3] {
        [self.pts[3 * q], self.pts[3 * q + 1], self.pts[3 * q + 2]]
    }
    fn normal(&self, q: usize) -> [f64; 3] {
        [self.nrm[3 * q], self.nrm[3 * q + 1], self.nrm[3 * q + 2]]
    }
    /// The generalized normals of every body point: its body's modes in body axes about its
    /// rotation centre, then its generalized modes (n·S_g; the other bodies' columns not stored).
    pub(super) fn generalized_normals(&self, m: &Problem) -> Vec<Vec<f64>> {
        (0..self.len())
            .map(|q| {
                let b = &m.bodies[self.owner[q]];
                let rc = b.rotation_center;
                let x = self.point(q);
                let xb = to_body(b, [x[0] - b.position[0], x[1] - b.position[1], x[2] - b.position[2]]);
                let nb = to_body(b, self.normal(q));
                let d = [xb[0] - rc[0], xb[1] - rc[1], xb[2] - rc[2]];
                let mut nj = rigid_normals(&d, &nb).to_vec();
                for gm in &b.generalized {
                    let s = gm.displacement(&v3(xb[0], xb[1], xb[2]), self.rec[q]);
                    nj.push(s[0] * nb[0] + s[1] * nb[1] + s[2] * nb[2]);
                }
                nj
            })
            .collect()
    }
}

/// A waterline segment t in the direction of Stokes's theorem for the hull of normal n: the hull
/// (below the waterline) on the left seen along n, (n × t)_z < 0.
fn along_waterline(n: &[f64; 3], t: [f64; 3]) -> [f64; 3] {
    if n[0] * t[1] - n[1] * t[0] > 0.0 { [-t[0], -t[1], -t[2]] } else { t }
}

/// A waterline segment: its midpoint and direction t (Stokes's theorem), the panel q next to it and
/// the next one q2 further from it, the values there extrapolated linearly from the two,
/// (1 + c) f_q − c f_q2 (from the panel next to it alone the transform of the second derivatives
/// would lose an order: its surface and waterline parts nearly cancel).
pub(super) struct Wl {
    pub(super) mid: [f64; 3],
    pub(super) t: [f64; 3],
    q: usize,
    q2: usize,
    c: f64,
}

impl Wl {
    /// The side of midpoint mid and direction t from the points d (normals n) that `wet` admits (the
    /// wetted panels: not the elements of a plate, whose edge may lie on the side too; all when
    /// empty).
    fn new(d: &[[f64; 3]], n: &[[f64; 3]], wet: &[bool], mid: [f64; 3], t: [f64; 3]) -> Wl {
        let ok = |q: usize| wet.is_empty() || wet[q];
        let dist =
            |a: &[f64; 3], b: &[f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        let nearest = |x: &[f64; 3], skip: usize| {
            let mut best = (f64::INFINITY, 0usize);
            for (q, dq) in d.iter().enumerate() {
                let e = dist(dq, x);
                if q != skip && ok(q) && e < best.0 {
                    best = (e, q);
                }
            }
            best.1
        };
        let q = nearest(&mid, usize::MAX);
        // the next point away from the waterline: on the ray from the midpoint through q, one to
        // three times as far again, the nearest to that ray (on a grid of output points that is
        // the point below q, not its neighbour along the waterline)
        let s1 = dist(&d[q], &mid);
        let u = [(d[q][0] - mid[0]) / s1, (d[q][1] - mid[1]) / s1, (d[q][2] - mid[2]) / s1];
        let mut best = (f64::INFINITY, q);
        for (k, dk) in d.iter().enumerate() {
            let r = [dk[0] - d[q][0], dk[1] - d[q][1], dk[2] - d[q][2]];
            let t = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
            if k == q || !ok(k) || t < 0.5 * s1 || t > 3.0 * s1 {
                continue;
            }
            let off = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - t * t).max(0.0).sqrt();
            if off < best.0 {
                best = (off, k);
            }
        }
        let q2 = best.1;
        let s2 = if q2 == q { s1 } else { dist(&d[q2], &mid) };
        let (q2, c) = if q2 != q && best.0 < 0.5 * s1 && s2 > 1.5 * s1 { (q2, s1 / (s2 - s1)) } else { (q, 0.0) };
        Wl { mid, t: along_waterline(&n[q], t), q, q2, c }
    }
    fn at(&self, v: &[C3]) -> C3 {
        std::array::from_fn(|k| (1.0 + self.c) * v[self.q][k] - self.c * v[self.q2][k])
    }
    fn at1(&self, f: impl Fn(usize) -> Complex64) -> Complex64 {
        (1.0 + self.c) * f(self.q) - self.c * f(self.q2)
    }
}

/// A segment of a body waterline for the line integral of the free surface: its midpoint (global),
/// length, horizontal normal into the body, the body panels it takes its values from (extrapolated
/// as `Wl`), and the total first-order potential and gradient of every (frequency, heading) there.
pub(super) struct WlFs {
    pub(super) x: [f64; 3],
    pub(super) dl: f64,
    pub(super) nu: [f64; 2],
    q: usize,
    q2: usize,
    c: f64,
    phi: Vec<Complex64>, // [F][H] total potential
    dnu: Vec<Complex64>, // [F][H] its derivative along ν
    nh: usize,
}

impl WlFs {
    /// The disturbance (total minus incident) of (f, h) with the incident values at the midpoint:
    /// its potential and a horizontal gradient of the right component along ν (the line integral
    /// needs only that one).
    pub(super) fn disturbance(&self, f: usize, hh: usize, pi: &Complex64, gi: &[Complex64; 3]) -> (Complex64, C3) {
        let k = f * self.nh + hh;
        let d = self.dnu[k] - (gi[0] * self.nu[0] + gi[1] * self.nu[1]);
        (self.phi[k] - pi, [d * self.nu[0], d * self.nu[1], CZ])
    }
    /// ψ at the midpoint from its values at the body points.
    pub(super) fn psi(&self, psi_b: impl Fn(usize) -> Complex64) -> Complex64 {
        (1.0 + self.c) * psi_b(self.q) - self.c * psi_b(self.q2)
    }
}

/// The waterline segments segs (global; waterline_segments) with the total first-order potential
/// extrapolated from the panels next to them (the diffraction and the radiation of the motions)
/// and its derivative along ν from the body condition and the free-surface condition, with n the
/// body normal into the fluid and ν = −n_h/|n_h|: ∂_ν φ = −(∂_n φ − n_z ∂_z φ)/|n_h|, ∂_n φ =
/// iω Σ ξ_k n_k, ∂_z φ = Kφ (no velocity of the source formulation, which converges only linearly).
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn free_surface_waterline(
    m: &Problem,
    ra: &Results,
    lay: &Layout,
    segs: &[WlSeg],
    omegas: &[f64],
    xi: &[Complex64],
    H: usize,
) -> Result<Vec<WlFs>> {
    let g = m.env.gravity;
    let F = omegas.len();
    let n6 = lay.nl;
    let Nb = lay.len();
    let nc = lay.nc;
    let bpd = read_c32(ra, QuantityId::BodyPointsPressureDiffraction); // [F][Nb][H]
    let bpr = read_c32(ra, QuantityId::BodyPointsPressureRadiation); // [F][Nb][nc]
    if bpd.len() != F * Nb * H || bpr.len() != F * Nb * nc {
        return Err(Error::internal("potential QTF: unexpected sizes of the body results"));
    }
    let mut out = Vec::with_capacity(segs.len());
    for (bi, b) in m.bodies.iter().enumerate() {
        let idx: Vec<usize> = (0..Nb).filter(|&q| lay.owner[q] == bi).collect();
        let d: Vec<[f64; 3]> = idx.iter().map(|&q| lay.point(q)).collect();
        let n: Vec<[f64; 3]> = idx.iter().map(|&q| lay.normal(q)).collect();
        let wet: Vec<bool> = idx.iter().map(|&q| !lay.plate[q]).collect();
        for s in segs.iter().filter(|s| s.body == bi) {
            let (p0, p1) = (s.p[0], s.p[1]);
            let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), 0.0];
            let t = [p1[0] - p0[0], p1[1] - p0[1], 0.0];
            let len = (t[0] * t[0] + t[1] * t[1]).sqrt();
            let w = Wl::new(&d, &n, &wet, mid, t);
            // the horizontal normal of the segment, into the body (against the normal into the fluid)
            let mut nu = [t[1] / len, -t[0] / len];
            if nu[0] * n[w.q][0] + nu[1] * n[w.q][1] > 0.0 {
                nu = [-nu[0], -nu[1]];
            }
            let (q, q2) = (idx[w.q], idx[w.q2]);
            // the generalized normals of the rigid modes at the midpoint, in body axes about the
            // rotation centre (n into the fluid, that of the panel next to the segment)
            let nq = n[w.q];
            let rc = b.rotation_center;
            let xb = to_body(b, [mid[0] - b.position[0], mid[1] - b.position[1], -b.position[2]]);
            let nb = to_body(b, nq);
            let nk = rigid_normals(&[xb[0] - rc[0], xb[1] - rc[1], xb[2] - rc[2]], &nb);
            let nhz = (1.0 - nq[2] * nq[2]).max(1e-6).sqrt();
            // the generalized modes' displacement at the midpoint (body axes, about O), on the
            // element of the panel next to it
            let xl = v3(xb[0], xb[1], xb[2]);
            let gs: Vec<Vec3> = b.generalized.iter().map(|gm| gm.displacement(&xl, lay.rec[q])).collect();
            let mut phi = Vec::with_capacity(F * H);
            let mut dnu = Vec::with_capacity(F * H);
            for f in 0..F {
                let toph = I * g / omegas[f];
                for hh in 0..H {
                    let at = |q: usize| -> Complex64 {
                        let mut p = bpd[(f * Nb + q) * H + hh];
                        for fr in &lay.free {
                            for &(j, cc) in fr {
                                p += xi[(f * H + hh) * n6 + j] * bpr[(f * Nb + q) * nc + cc];
                            }
                        }
                        p * toph
                    };
                    let ph = (1.0 + w.c) * at(q) - w.c * at(q2);
                    let mut dn = CZ;
                    for k in 0..6 {
                        dn += I * omegas[f] * xi[(f * H + hh) * n6 + lay.off[bi] + k] * nk[k];
                    }
                    for (k, s) in gs.iter().enumerate() {
                        let ng = s[0] * nb[0] + s[1] * nb[1] + s[2] * nb[2];
                        dn += I * omegas[f] * xi[(f * H + hh) * n6 + lay.off[bi] + 6 + k] * ng;
                    }
                    let kk = omegas[f] * omegas[f] / g;
                    phi.push(ph);
                    dnu.push(-(dn - nq[2] * kk * ph) / nhz);
                }
            }
            out.push(WlFs { x: mid, dl: len, nu, q, q2, c: w.c, phi, dnu, nh: H });
        }
    }
    Ok(out)
}

/// The first-order state of a wave on the body: the displacement X = ξ + α × x (+ Σ ξ_g S_g of the
/// generalized modes) of the points, α, ∇φ of the total first-order potential at the panels and at
/// the waterline segments, with ω; with generalized modes their displacement gradient
/// Eg = Σ ξ_g ∇S_g at the panels (empty for a rigid body).
pub(super) struct State {
    w: Complex64, // iω (−iω for the conjugate)
    alpha: C3,
    x: Vec<C3>,
    grad: Vec<C3>,
    xw: Vec<C3>,
    gradw: Vec<C3>,
    eg: Vec<[C3; 3]>,
}

impl State {
    fn conj(&self) -> State {
        State {
            w: self.w.conj(),
            alpha: conj3(&self.alpha),
            x: self.x.iter().map(conj3).collect(),
            grad: self.grad.iter().map(conj3).collect(),
            xw: self.xw.iter().map(conj3).collect(),
            gradw: self.gradw.iter().map(conj3).collect(),
            eg: self.eg.iter().map(|e| [conj3(&e[0]), conj3(&e[1]), conj3(&e[2])]).collect(),
        }
    }
    /// The first-order change of the unit normal n at point q: α × n, and of a displacement
    /// gradient Eg its tangential part −(Egᵀ n − (n·Egᵀ n) n).
    fn turned_normal(&self, q: usize, n: &[f64; 3]) -> C3 {
        let mut d = crossr(&self.alpha, n);
        if let Some(e) = self.eg.get(q) {
            let etn: C3 = std::array::from_fn(|k| e[0][k] * n[0] + e[1][k] * n[1] + e[2][k] * n[2]);
            let nn = dotr(&etn, n);
            for k in 0..3 {
                d[k] -= etn[k] - nn * n[k];
            }
        }
        d
    }
    /// The terms of the second derivatives that the transform leaves on the surface beyond the
    /// rigid ones: with X of displacement gradient E = A + Eg, (X·∇)∇φ = ∇ × (∇φ × X) − (∇·X)∇φ +
    /// E∇φ, the rigid part n·(α × ∇φ) and of Eg n·(Eg ∇φ) − (tr Eg)(n·∇φ), for ∇φ = gp.
    fn deformation(&self, q: usize, n: &[f64; 3], gp: &C3) -> Complex64 {
        let Some(e) = self.eg.get(q) else { return CZ };
        let eg: C3 = std::array::from_fn(|i| dot(&e[i], gp));
        dotr(&eg, n) - (e[0][0] + e[1][1] + e[2][2]) * dotr(gp, n)
    }
}

/// The motion terms of Q_B of a pair as the integral ∫ G Q_B dS of a function G on the body (ψ_j
/// in the indirect method, the Green function of a collocation point in the direct one, theory
/// §12.7): Σ_q G_q u_q A_q + Σ_q ∇G_q·T_q + Σ_w G_w l_w, per body point the scalar part u (the
/// second rotation, the turned normal, G n·(α × ∇φ) of the Stokes transform), the vector T (times
/// the area, body axes; tangential: n·T = 0) and per waterline segment its factor l.
pub(super) struct PairTerms {
    pub(super) u: Vec<Complex64>,
    pub(super) t: Vec<C3>,
    pub(super) l: Vec<Complex64>,
}

/// The body forcing of the first-order motions of one body (Lee 1995, eqs 3.11-3.15), in its body
/// axes about its rotation centre: its panels (d = x − x_R, n, area), its waterline segments (the
/// midpoint, the panel next to it, dl oriented with the hull on its left seen along n), ψ_j and
/// ∇ψ_j at Ω of the n6 coordinates of all bodies there (the indirect method).
pub(super) struct Motion {
    pub(super) bi: usize,       // the body
    pub(super) idx: Vec<usize>, // its body points among all
    d: Vec<[f64; 3]>,
    n: Vec<[f64; 3]>,
    area: Vec<f64>,
    pub(super) wl: Vec<Wl>,
    pub(super) free: Vec<bool>, // a waterline side by the free surface (else under a plate)
    states: Vec<State>,         // [F][H]
    nh: usize,
    n6: usize,
    psi: Vec<Complex64>,
    gpsi: Vec<C3>, // [O][this body's points][n6]
}

/// The plate over a waterline side of a body (theory §12.7, a plate whose edge lies on a wall):
/// its body pb, the side's midpoint in pb's axes about pb's rotation centre, the displacements of
/// pb's generalized modes there on the plate's element (pb's axes about its origin), and the turn
/// (cos, sin) from pb's axes to those of the side's body.
struct Over {
    pb: usize,
    d: [f64; 3],
    s: Vec<Vec3>,
    turn: (f64, f64),
}

/// Whether body bi needs the motion terms of the body condition: it moves, or a waterline side of
/// it lies under a plate of another body that moves (the line integral of the plate's edge, which
/// moves against the wall: theory §12.7).
pub(super) fn has_motion_terms(m: &Problem, bi: usize, sides: &[BodySide]) -> bool {
    let moves = |b: &crate::problem::Body| !b.modes.is_empty() || !b.generalized.is_empty();
    moves(&m.bodies[bi]) || sides.iter().any(|s| s.plate.is_some_and(|(pb, _)| pb != bi && moves(&m.bodies[pb])))
}

impl Motion {
    /// The motion terms of body bi (its waterline sides, the zones' body_sides) from the first-order
    /// results ra with the motions xi [F][H][n6] of every body
    /// at the frequencies omegas; rb: the radiation at the pair frequencies oms (the indirect
    /// method's ψ; none for the direct method alone).
    #[allow(non_snake_case)] // the notation of the theory documents
    #[allow(clippy::needless_range_loop)] // index notation over several arrays
    #[allow(clippy::too_many_arguments)] // the body, both sub-problems, the motions, the frequencies
    pub(super) fn new(
        m: &Problem,
        bi: usize,
        sides: &[BodySide],
        lay: &Layout,
        ra: &Results,
        rb: Option<&Results>,
        xi: &[Complex64],
        omegas: &[f64],
        oms: &[f64],
        H: usize,
    ) -> Result<Motion> {
        let g = m.env.gravity;
        let b = &m.bodies[bi];
        let F = omegas.len();
        let NO = oms.len();
        let n6 = lay.nl;
        let idx: Vec<usize> = (0..lay.len()).filter(|&q| lay.owner[q] == bi).collect();
        let (cy, sy) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
        let to_body_c = |v: C3| [v[0] * cy + v[1] * sy, -v[0] * sy + v[1] * cy, v[2]];
        let rc = b.rotation_center;
        let Nb = lay.len();
        let area: Vec<f64> = idx.iter().map(|&q| lay.area[q]).collect();
        let mut d = Vec::with_capacity(idx.len());
        let mut n = Vec::with_capacity(idx.len());
        for &q in &idx {
            let p = lay.point(q);
            let x = to_body(b, [p[0] - b.position[0], p[1] - b.position[1], p[2] - b.position[2]]);
            d.push([x[0] - rc[0], x[1] - rc[1], x[2] - rc[2]]);
            n.push(to_body(b, lay.normal(q)));
        }
        // the waterline: the panel edges in z = 0 of the mesh (body axes), reflected as the mesh
        // (in the order of the zones' sides), its values from the wetted panels next to it
        let wet: Vec<bool> = idx.iter().map(|&q| !lay.plate[q]).collect();
        let wl: Vec<Wl> = sides
            .iter()
            .map(|x| {
                let s = &x.local;
                let mid = [
                    0.5 * (s[0][0] + s[1][0]) - rc[0],
                    0.5 * (s[0][1] + s[1][1]) - rc[1],
                    0.5 * (s[0][2] + s[1][2]) - rc[2],
                ];
                Wl::new(&d, &n, &wet, mid, [s[1][0] - s[0][0], s[1][1] - s[0][1], s[1][2] - s[0][2]])
            })
            .collect();
        let free: Vec<bool> = sides.iter().map(|x| x.plate.is_none()).collect();
        // the plates over sides of the waterline: the plate's edge on the wall, its line integral
        // that of the side with the plate's displacement (theory §12.7)
        let over: Vec<Option<Over>> = sides
            .iter()
            .zip(&wl)
            .map(|(x, w)| {
                let (pb, e) = x.plate?;
                let p = &m.bodies[pb];
                // the midpoint: body axes about bi's centre -> global -> pb's axes
                let xb = [w.mid[0] + rc[0], w.mid[1] + rc[1], w.mid[2] + rc[2]];
                let g = [b.position[0] + cy * xb[0] - sy * xb[1], b.position[1] + sy * xb[0] + cy * xb[1], 0.0];
                let xp = to_body(p, [g[0] - p.position[0], g[1] - p.position[1], g[2] - p.position[2]]);
                let rp = p.rotation_center;
                let turn = ((p.yaw - b.yaw) * PI / 180.0).sin_cos();
                Some(Over {
                    pb,
                    d: [xp[0] - rp[0], xp[1] - rp[1], xp[2] - rp[2]],
                    s: p.generalized.iter().map(|gm| gm.displacement(&v3(xp[0], xp[1], xp[2]), e)).collect(),
                    turn: (turn.1, turn.0),
                })
            })
            .collect();
        // the generalized modes: their displacement and gradient at the points (body axes about O)
        // and their displacement at the waterline midpoints, on the element of the panel next to it
        let gmod = &b.generalized;
        let at_o = |p: &[f64; 3]| v3(p[0] + rc[0], p[1] + rc[1], p[2] + rc[2]);
        let gs: Vec<Vec<Vec3>> = idx
            .iter()
            .zip(&d)
            .map(|(&q, p)| gmod.iter().map(|gm| gm.displacement(&at_o(p), lay.rec[q])).collect())
            .collect();
        let gb: Vec<Vec<[[f64; 3]; 3]>> = idx
            .iter()
            .zip(&d)
            .map(|(&q, p)| gmod.iter().map(|gm| gm.gradient(&at_o(p), lay.rec[q])).collect())
            .collect();
        let gw: Vec<Vec<Vec3>> = wl
            .iter()
            .map(|w| gmod.iter().map(|gm| gm.displacement(&at_o(&w.mid), lay.rec[idx[w.q]])).collect())
            .collect();
        // the first-order states
        let nc = lay.nc; // the columns of all bodies
        let bvd = read_c32(ra, QuantityId::BodyPointsVelocityDiffraction); // [F][Nb][H][3]
        let bvr = read_c32(ra, QuantityId::BodyPointsVelocityRadiation); // [F][Nb][nc][3]
        // ψ at Ω (the indirect method only)
        let (bpr, bvo) = match rb {
            Some(rb) => (
                read_c32(rb, QuantityId::BodyPointsPressureRadiation),
                read_c32(rb, QuantityId::BodyPointsVelocityRadiation),
            ),
            None => (Vec::new(), Vec::new()),
        }; // [O][Nb][n6], [O][Nb][n6][3]
        if bvd.len() != 3 * F * Nb * H
            || bvr.len() != 3 * F * Nb * nc
            || (rb.is_some() && (bpr.len() != NO * Nb * n6 || bvo.len() != 3 * NO * Nb * n6))
        {
            return Err(Error::internal("potential QTF: unexpected sizes of the body results"));
        }
        let mut states = Vec::with_capacity(F * H);
        for f in 0..F {
            for hh in 0..H {
                let o = (f * H + hh) * n6 + lay.off[bi];
                let x6 = &xi[o..o + 6];
                let xg = &xi[o + 6..o + 6 + gmod.len()];
                let alpha = [x6[3], x6[4], x6[5]];
                let disp = |p: &[f64; 3], s: &[Vec3]| -> C3 {
                    let c = crossr(&alpha, p);
                    let mut u = [x6[0] + c[0], x6[1] + c[1], x6[2] + c[2]];
                    for (x, sk) in xg.iter().zip(s) {
                        for k in 0..3 {
                            u[k] += x * sk[k];
                        }
                    }
                    u
                };
                let mut grad = Vec::with_capacity(idx.len());
                for &q in &idx {
                    let mut v: C3 = std::array::from_fn(|k| bvd[((f * Nb + q) * H + hh) * 3 + k]);
                    // the radiation of every moving body's motions
                    for fr in &lay.free {
                        for &(j, cc) in fr {
                            let x = xi[(f * H + hh) * n6 + j];
                            for k in 0..3 {
                                v[k] += x * bvr[((f * Nb + q) * nc + cc) * 3 + k];
                            }
                        }
                    }
                    grad.push(to_body_c(v));
                }
                let eg: Vec<[C3; 3]> = if gmod.is_empty() {
                    Vec::new()
                } else {
                    gb.iter()
                        .map(|bq| {
                            std::array::from_fn(|i| {
                                std::array::from_fn(|l| xg.iter().zip(bq).map(|(x, bk)| x * bk[i][l]).sum())
                            })
                        })
                        .collect()
                };
                // the displacement at a side: the wall's, less the plate's over it (the line
                // integrals of the wall's and of the plate's transforms along their common edge, run
                // the opposite ways: the jump of the displacement there; zero for a plate moving
                // with the wall)
                let xw: Vec<C3> = wl
                    .iter()
                    .zip(&gw)
                    .zip(&over)
                    .map(|((w, s), ov)| {
                        let mut u = disp(&w.mid, s);
                        if let Some(ov) = ov {
                            let o = (f * H + hh) * n6 + lay.off[ov.pb];
                            let (p6, pg) = (&xi[o..o + 6], &xi[o + 6..o + 6 + ov.s.len()]);
                            let c = crossr(&[p6[3], p6[4], p6[5]], &ov.d);
                            let mut v = [p6[0] + c[0], p6[1] + c[1], p6[2] + c[2]];
                            for (x, sk) in pg.iter().zip(&ov.s) {
                                for k in 0..3 {
                                    v[k] += x * sk[k];
                                }
                            }
                            let (ct, st) = ov.turn;
                            let v = [ct * v[0] - st * v[1], st * v[0] + ct * v[1], v[2]];
                            for k in 0..3 {
                                u[k] -= v[k];
                            }
                        }
                        u
                    })
                    .collect();
                states.push(State {
                    w: Complex64::new(0.0, omegas[f]),
                    alpha,
                    x: d.iter().zip(&gs).map(|(p, s)| disp(p, s)).collect(),
                    xw,
                    gradw: wl.iter().map(|w| w.at(&grad)).collect(),
                    grad,
                    eg,
                });
            }
        }
        // ψ (per unit velocity) and ∇ψ = v / (iΩ) in this body's axes
        let nq = idx.len();
        let no = if rb.is_some() { NO } else { 0 };
        let mut psi = vec![CZ; no * nq * n6];
        let mut gpsi = vec![[CZ; 3]; no * nq * n6];
        for o in 0..no {
            for (k, &q) in idx.iter().enumerate() {
                for j in 0..n6 {
                    let (i, s) = ((o * nq + k) * n6 + j, (o * Nb + q) * n6 + j);
                    psi[i] = bpr[s] * (g / (oms[o] * oms[o]));
                    let v: C3 = std::array::from_fn(|c| bvo[s * 3 + c] / (I * oms[o]));
                    gpsi[i] = to_body_c(v);
                }
            }
        }
        Ok(Motion { bi, idx, d, n, area, wl, free, states, nh: H, n6, psi, gpsi })
    }

    /// The motion terms of Q_B of the pair (a, ha; b, hb), Ω = ω_a ± ω_b (PairTerms): the second
    /// rotation (H), the turned normal and the second derivatives by Stokes's theorem (Lee 1995,
    /// eqs 3.11-3.15, eq 3.30 with G in place of the Green function: ∫ G n·[(X·∇)∇φ] dS =
    /// ∫ (n·X)(∇φ·∇G) + G n·(α × ∇φ) − (n·∇φ)(X·∇G) dS + ∮ G (∇φ × X)·dl, with generalized modes
    /// also G [n·(Eg ∇φ) − (tr Eg)(n·∇φ)], State::deformation), both orders of the pair.
    #[allow(clippy::needless_range_loop)] // index notation over several arrays
    pub(super) fn pair_terms(&self, a: usize, ha: usize, b: usize, hb: usize, sum: bool) -> PairTerms {
        let nh = self.nh;
        let sa = &self.states[a * nh + ha];
        let conj;
        let sb = if sum {
            &self.states[b * nh + hb]
        } else {
            conj = self.states[b * nh + hb].conj();
            &conj
        };
        let om = (sa.w + sb.w) / I; // Ω
        let (p, c) = (sa.alpha, sb.alpha);
        // the second rotation: H x (Lee 1995, eqs 3.14, 3.15, with its factor 1/2)
        let hm = [
            [-(p[1] * c[1] + p[2] * c[2]), CZ, CZ],
            [p[0] * c[1] + c[0] * p[1], -(p[0] * c[0] + p[2] * c[2]), CZ],
            [p[0] * c[2] + c[0] * p[2], p[1] * c[2] + c[1] * p[2], -(p[0] * c[0] + p[1] * c[1])],
        ];
        let nb = self.area.len();
        let mut u = vec![CZ; nb];
        let mut t = vec![[CZ; 3]; nb];
        for q in 0..nb {
            let (d, n) = (&self.d[q], &self.n[q]);
            let hx: C3 = std::array::from_fn(|r| 0.5 * (hm[r][0] * d[0] + hm[r][1] * d[1] + hm[r][2] * d[2]));
            let t1 = 0.5 * I * om * dotr(&hx, n);
            let va: C3 = std::array::from_fn(|k| sa.w * sa.x[q][k] - sa.grad[q][k]);
            let vb: C3 = std::array::from_fn(|k| sb.w * sb.x[q][k] - sb.grad[q][k]);
            // the turned normal (with generalized modes the deformation's too)
            let t2 = 0.25 * (dot(&sa.turned_normal(q, n), &vb) + dot(&sb.turned_normal(q, n), &va));
            // the Stokes transform of both orders, times −1/4: G n·(α_x × ∇φ) (and the deformation)
            let mut ts = dotr(&cross(&sa.alpha, &sb.grad[q]), n) + dotr(&cross(&sb.alpha, &sa.grad[q]), n);
            if !sa.eg.is_empty() {
                ts += sa.deformation(q, n, &sb.grad[q]) + sb.deformation(q, n, &sa.grad[q]);
            }
            u[q] = t1 + t2 - 0.25 * ts;
            // and (n·X_x)(∇φ·∇G) − (n·∇φ)(X_x·∇G)
            let (nxa, nxb) = (dotr(&sa.x[q], n), dotr(&sb.x[q], n));
            let (nga, ngb) = (dotr(&sa.grad[q], n), dotr(&sb.grad[q], n));
            for k in 0..3 {
                let v = nxa * sb.grad[q][k] - ngb * sa.x[q][k] + nxb * sa.grad[q][k] - nga * sb.x[q][k];
                t[q][k] = -0.25 * v * self.area[q];
            }
        }
        // the waterline: ∮ G (∇φ × X)·dl of both orders, times −1/4
        let l = (0..self.wl.len())
            .map(|k| {
                let w = &self.wl[k];
                -0.25 * (dotr(&cross(&sb.gradw[k], &sa.xw[k]), &w.t) + dotr(&cross(&sa.gradw[k], &sb.xw[k]), &w.t))
            })
            .collect();
        PairTerms { u, t, l }
    }

    /// The direct method's line integral of the sides under a plate (no field point of the free
    /// surface there): G at the side extrapolated from the wetted panels as ψ is (Wl), so their
    /// factors l join the scalar part u of those panels (per area): (body point, its term).
    pub(super) fn under_plates(&self, pt: &PairTerms) -> Vec<(usize, Complex64)> {
        let mut out = Vec::new();
        for ((w, l), &free) in self.wl.iter().zip(&pt.l).zip(&self.free) {
            if !free {
                out.push((self.idx[w.q], (1.0 + w.c) * l / self.area[w.q]));
                if w.c != 0.0 {
                    out.push((self.idx[w.q2], -w.c * l / self.area[w.q2]));
                }
            }
        }
        out
    }

    /// ∫ ψ_j (the motion terms of Q_B) dS of a pair from its terms (pair_terms) at the pair
    /// frequency o: Σ_q ψ u A + Σ_q ∇ψ·T + Σ_w ψ_w l, ψ_w extrapolated to the waterline (Wl).
    pub(super) fn body_term(&self, pt: &PairTerms, o: usize, j: usize) -> Complex64 {
        let (nb, n6) = (self.area.len(), self.n6);
        let mut s = CZ;
        for q in 0..nb {
            let i = (o * nb + q) * n6 + j;
            s += self.psi[i] * pt.u[q] * self.area[q] + dot(&self.gpsi[i], &pt.t[q]);
        }
        for (w, l) in self.wl.iter().zip(&pt.l) {
            s += w.at1(|q| self.psi[(o * nb + q) * n6 + j]) * l;
        }
        s
    }

    /// ∫ ψ n·[(X·∇)∇φ] dS by Stokes's theorem (Lee 1995, eq 3.30 with ψ in place of G):
    /// ∫ (n·X)(∇φ·∇ψ) + ψ n·(α × ∇φ) − (n·∇φ)(X·∇ψ) dS + ∮ ψ (∇φ × X)·dl, with generalized modes
    /// also ψ [n·(Eg ∇φ) − (tr Eg)(n·∇φ)] (State::deformation); the transform pair_terms takes,
    /// for one order of the pair, here for its test.
    #[cfg(test)]
    fn stokes(&self, x: &State, phi: &State, o: usize, j: usize) -> Complex64 {
        let (nb, n6) = (self.area.len(), self.n6);
        let mut s = CZ;
        for q in 0..nb {
            let i = (o * nb + q) * n6 + j;
            let n = &self.n[q];
            let (gp, gs) = (&phi.grad[q], &self.gpsi[i]);
            s += (dotr(&x.x[q], n) * dot(gp, gs) + self.psi[i] * dotr(&cross(&x.alpha, gp), n)
                - dotr(gp, n) * dot(&x.x[q], gs))
                * self.area[q];
            if !x.eg.is_empty() {
                s += self.psi[i] * x.deformation(q, n, gp) * self.area[q];
            }
        }
        for (k, w) in self.wl.iter().enumerate() {
            let psi = w.at1(|q| self.psi[(o * nb + q) * n6 + j]);
            s += psi * dotr(&cross(&phi.gradw[k], &x.xw[k]), &w.t);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_types::wavenumber;

    #[test]
    fn stokes_on_a_hemisphere() {
        // ∫ ψ n·[(X·∇)∇φ] dS by the transform against the Hessian of φ, on the hemisphere r = 1
        // (normals into the body, midpoint rule in θ and the polar angle), φ and ψ incident waves
        // of depth 3 at 1.5 and 2.3 rad/s (headings 0 and 30°), X a rigid displacement
        let (g, h) = (9.80665, 3.0);
        let inc = |w: f64, beta: f64, x: &[f64; 3]| -> (Complex64, C3, [C3; 3]) {
            let k = wavenumber(w, h, g).unwrap();
            let kv = [-I * k * beta.cos(), -I * k * beta.sin()];
            let e = I * g / w * Complex64::from_polar(1.0, -k * (x[0] * beta.cos() + x[1] * beta.sin()));
            let (z, zz) = ((k * (x[2] + h)).cosh() / (k * h).cosh(), k * (k * (x[2] + h)).sinh() / (k * h).cosh());
            let phi = e * z;
            let gr = [kv[0] * phi, kv[1] * phi, e * zz];
            let hs = [
                [kv[0] * kv[0] * phi, kv[0] * kv[1] * phi, kv[0] * e * zz],
                [kv[1] * kv[0] * phi, kv[1] * kv[1] * phi, kv[1] * e * zz],
                [kv[0] * e * zz, kv[1] * e * zz, k * k * phi],
            ];
            (phi, gr, hs)
        };
        let (nt, np) = (192, 96);
        let xi = [Complex64::new(0.3, 0.1), Complex64::new(-0.2, 0.0), Complex64::new(0.0, 0.5)];
        let alpha = [Complex64::new(0.1, 0.0), Complex64::new(0.0, 0.2), Complex64::new(-0.3, 0.0)];
        let disp = |p: &[f64; 3]| -> C3 {
            let c = crossr(&alpha, p);
            [xi[0] + c[0], xi[1] + c[1], xi[2] + c[2]]
        };
        let (mut d, mut n, mut area, mut psi, mut gpsi, mut grad) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        let mut lhs = CZ;
        for i in 0..nt {
            let th = 2.0 * PI * (i as f64 + 0.5) / nt as f64;
            for j in 0..np {
                let (p0, p1) = (0.5 * PI * j as f64 / np as f64, 0.5 * PI * (j + 1) as f64 / np as f64);
                let pm = 0.5 * (p0 + p1); // the angle below the waterline
                let x = [pm.cos() * th.cos(), pm.cos() * th.sin(), -pm.sin()];
                let a = (p1.sin() - p0.sin()) * 2.0 * PI / nt as f64;
                let (_, gp, hp) = inc(1.5, 0.0, &x);
                let (ps, gs, _) = inc(2.3, PI / 6.0, &x);
                let nn = [-x[0], -x[1], -x[2]];
                let xd = disp(&x);
                let hx: C3 = std::array::from_fn(|r| hp[r][0] * xd[0] + hp[r][1] * xd[1] + hp[r][2] * xd[2]);
                lhs += ps * dotr(&hx, &nn) * a;
                d.push(x);
                n.push(nn);
                area.push(a);
                grad.push(gp);
                for _ in 0..6 {
                    psi.push(ps);
                    gpsi.push(gs);
                }
            }
        }
        // the waterline: the chords of the top row, given either way round; ψ and ∇φ there
        // extrapolated from the two rows next to it
        let mut wl = Vec::new();
        let mut xw = vec![];
        for i in 0..nt {
            let (t0, t1) = (2.0 * PI * i as f64 / nt as f64, 2.0 * PI * (i + 1) as f64 / nt as f64);
            let (a, b) = ([t0.cos(), t0.sin(), 0.0], [t1.cos(), t1.sin(), 0.0]);
            let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0];
            let t = if i % 2 == 0 { [b[0] - a[0], b[1] - a[1], 0.0] } else { [a[0] - b[0], a[1] - b[1], 0.0] };
            let w = Wl::new(&d, &n, &[], mid, t);
            assert!(w.q == i * np && w.q2 == i * np + 1 && w.c > 0.0);
            wl.push(w);
            xw.push(disp(&mid));
        }
        let gradw: Vec<C3> = wl.iter().map(|w| w.at(&grad)).collect();
        let x = State { w: CZ, alpha, x: d.iter().map(disp).collect(), grad: vec![], xw, gradw: vec![], eg: vec![] };
        let ph = State { w: CZ, alpha, x: vec![], grad, xw: vec![], gradw, eg: vec![] };
        let free = vec![true; wl.len()];
        let mut mo = Motion { bi: 0, idx: vec![], d, n, area, wl, free, states: vec![], nh: 1, n6: 6, psi, gpsi };
        let rhs = mo.stokes(&x, &ph, 0, 0);
        assert!((rhs - lhs).norm() <= 2e-3 * lhs.norm(), "{lhs} {rhs}");
        // the other way round the waterline is wrong
        for w in &mut mo.wl {
            w.t = [-w.t[0], -w.t[1], -w.t[2]];
        }
        let wrong = mo.stokes(&x, &ph, 0, 0);
        assert!((wrong - lhs).norm() > 0.1 * lhs.norm(), "{lhs} {wrong}");
    }

    #[test]
    fn stokes_on_a_cup_under_a_sliding_plate() {
        // ∫ ψ n·[(X·∇)∇φ] dS over the water inside a cup (the wall r = 1, -1 < z < 0, and its
        // bottom, displaced by X_w) under a plate in z = 0 (displaced by X_p), against the transform
        // on each surface with its own X and the one line integral of the wall's top side with the
        // jump X_w − X_p (the plate's line runs the other way along the same edge), ψ and ∇φ there
        // from the wall's panels (the plate's points masked out); normals into the water
        let (g, h) = (9.80665, 3.0);
        let inc = |w: f64, beta: f64, x: &[f64; 3]| -> (Complex64, C3, [C3; 3]) {
            let k = wavenumber(w, h, g).unwrap();
            let kv = [-I * k * beta.cos(), -I * k * beta.sin()];
            let e = I * g / w * Complex64::from_polar(1.0, -k * (x[0] * beta.cos() + x[1] * beta.sin()));
            let (z, zz) = ((k * (x[2] + h)).cosh() / (k * h).cosh(), k * (k * (x[2] + h)).sinh() / (k * h).cosh());
            let phi = e * z;
            let gr = [kv[0] * phi, kv[1] * phi, e * zz];
            let hs = [
                [kv[0] * kv[0] * phi, kv[0] * kv[1] * phi, kv[0] * e * zz],
                [kv[1] * kv[0] * phi, kv[1] * kv[1] * phi, kv[1] * e * zz],
                [kv[0] * e * zz, kv[1] * e * zz, k * k * phi],
            ];
            (phi, gr, hs)
        };
        let rigid = |xi: [Complex64; 3], al: [Complex64; 3]| {
            move |p: &[f64; 3]| -> C3 {
                let c = crossr(&al, p);
                [xi[0] + c[0], xi[1] + c[1], xi[2] + c[2]]
            }
        };
        let (aw, ap) = (
            [Complex64::new(0.1, 0.0), Complex64::new(0.0, 0.2), Complex64::new(-0.3, 0.0)],
            [Complex64::new(0.0, -0.2), Complex64::new(0.15, 0.0), Complex64::new(0.0, 0.1)],
        );
        let xw = rigid([Complex64::new(0.3, 0.1), Complex64::new(-0.2, 0.0), Complex64::new(0.0, 0.5)], aw);
        let xp = rigid([Complex64::new(-0.1, 0.0), Complex64::new(0.2, 0.3), Complex64::new(0.4, -0.2)], ap);
        let (nt, nz) = (192, 96);
        let (mut d, mut n, mut wet, mut grad, mut psi) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut lhs, mut rhs) = (CZ, CZ);
        let mut point = |x: [f64; 3], nn: [f64; 3], a: f64, plate: bool| {
            let (_, gp, hp) = inc(1.5, 0.0, &x);
            let (ps, gs, _) = inc(2.3, PI / 6.0, &x);
            let (xd, al) = if plate { (xp(&x), ap) } else { (xw(&x), aw) };
            let hx: C3 = std::array::from_fn(|r| hp[r][0] * xd[0] + hp[r][1] * xd[1] + hp[r][2] * xd[2]);
            lhs += ps * dotr(&hx, &nn) * a;
            rhs += (dotr(&xd, &nn) * dot(&gp, &gs) + ps * dotr(&cross(&al, &gp), &nn) - dotr(&gp, &nn) * dot(&xd, &gs))
                * a;
            d.push(x);
            n.push(nn);
            wet.push(!plate);
            grad.push(gp);
            psi.push(ps);
        };
        for i in 0..nt {
            let (t0, t1) = (2.0 * PI * i as f64 / nt as f64, 2.0 * PI * (i + 1) as f64 / nt as f64);
            let th = 0.5 * (t0 + t1);
            let (c, s) = (th.cos(), th.sin());
            for j in 0..nz {
                // the wall, from the top down (its top row next to the waterline)
                let z = -(j as f64 + 0.5) / nz as f64;
                point([c, s, z], [-c, -s, 0.0], (t1 - t0) / nz as f64, false);
            }
            for j in 0..nz {
                // the bottom (normal up) and the plate (normal down), rings of equal width
                let (r0, r1) = (j as f64 / nz as f64, (j + 1) as f64 / nz as f64);
                let (r, a) = (0.5 * (r0 + r1), 0.5 * (r1 * r1 - r0 * r0) * (t1 - t0));
                point([r * c, r * s, -1.0], [0.0, 0.0, 1.0], a, false);
                point([r * c, r * s, 0.0], [0.0, 0.0, -1.0], a, true);
            }
        }
        // the top side of the wall, each segment given either way round (wrong: the wall's own line
        // alone, without the plate's)
        let mut wrong = rhs;
        for i in 0..nt {
            let (t0, t1) = (2.0 * PI * i as f64 / nt as f64, 2.0 * PI * (i + 1) as f64 / nt as f64);
            let (a, b) = ([t0.cos(), t0.sin(), 0.0], [t1.cos(), t1.sin(), 0.0]);
            let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0];
            let t = if i % 2 == 0 { [b[0] - a[0], b[1] - a[1], 0.0] } else { [a[0] - b[0], a[1] - b[1], 0.0] };
            let w = Wl::new(&d, &n, &wet, mid, t);
            assert!(wet[w.q] && wet[w.q2] && w.c > 0.0);
            let jump: C3 = std::array::from_fn(|k| xw(&mid)[k] - xp(&mid)[k]);
            rhs += w.at1(|q| psi[q]) * dotr(&cross(&w.at(&grad), &jump), &w.t);
            wrong += w.at1(|q| psi[q]) * dotr(&cross(&w.at(&grad), &xw(&mid)), &w.t);
        }
        assert!((rhs - lhs).norm() <= 2e-3 * lhs.norm(), "{lhs} {rhs}");
        assert!((wrong - lhs).norm() > 0.05 * lhs.norm(), "{lhs} {wrong}");
    }

    #[test]
    fn stokes_of_a_deformation_on_a_hemisphere() {
        // as stokes_on_a_hemisphere with a displacement that is not rigid: X = ξ + α × x + c u(x),
        // u = (x z, y², x y z) with divergence (z + 2y + x y), its gradient in the state; the
        // transform with the deformation terms of State::deformation against the Hessian of φ
        let (g, h) = (9.80665, 3.0);
        let inc = |w: f64, beta: f64, x: &[f64; 3]| -> (Complex64, C3, [C3; 3]) {
            let k = wavenumber(w, h, g).unwrap();
            let kv = [-I * k * beta.cos(), -I * k * beta.sin()];
            let e = I * g / w * Complex64::from_polar(1.0, -k * (x[0] * beta.cos() + x[1] * beta.sin()));
            let (z, zz) = ((k * (x[2] + h)).cosh() / (k * h).cosh(), k * (k * (x[2] + h)).sinh() / (k * h).cosh());
            let phi = e * z;
            let gr = [kv[0] * phi, kv[1] * phi, e * zz];
            let hs = [
                [kv[0] * kv[0] * phi, kv[0] * kv[1] * phi, kv[0] * e * zz],
                [kv[1] * kv[0] * phi, kv[1] * kv[1] * phi, kv[1] * e * zz],
                [kv[0] * e * zz, kv[1] * e * zz, k * k * phi],
            ];
            (phi, gr, hs)
        };
        let (nt, np) = (192, 96);
        let xi = [Complex64::new(0.3, 0.1), Complex64::new(-0.2, 0.0), Complex64::new(0.0, 0.5)];
        let alpha = [Complex64::new(0.1, 0.0), Complex64::new(0.0, 0.2), Complex64::new(-0.3, 0.0)];
        let c = Complex64::new(0.4, -0.3);
        let disp = |p: &[f64; 3]| -> C3 {
            let r = crossr(&alpha, p);
            let u = [p[0] * p[2], p[1] * p[1], p[0] * p[1] * p[2]];
            [xi[0] + r[0] + c * u[0], xi[1] + r[1] + c * u[1], xi[2] + r[2] + c * u[2]]
        };
        let grad_u = |p: &[f64; 3]| -> [C3; 3] {
            let r = |a: f64, b: f64, d: f64| [c * a, c * b, c * d];
            [r(p[2], 0.0, p[0]), r(0.0, 2.0 * p[1], 0.0), r(p[1] * p[2], p[0] * p[2], p[0] * p[1])]
        };
        let (mut d, mut n, mut area, mut psi, mut gpsi, mut grad, mut eg) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let mut lhs = CZ;
        for i in 0..nt {
            let th = 2.0 * PI * (i as f64 + 0.5) / nt as f64;
            for j in 0..np {
                let (p0, p1) = (0.5 * PI * j as f64 / np as f64, 0.5 * PI * (j + 1) as f64 / np as f64);
                let pm = 0.5 * (p0 + p1);
                let x = [pm.cos() * th.cos(), pm.cos() * th.sin(), -pm.sin()];
                let a = (p1.sin() - p0.sin()) * 2.0 * PI / nt as f64;
                let (_, gp, hp) = inc(1.5, 0.0, &x);
                let (ps, gs, _) = inc(2.3, PI / 6.0, &x);
                let nn = [-x[0], -x[1], -x[2]];
                let xd = disp(&x);
                let hx: C3 = std::array::from_fn(|r| hp[r][0] * xd[0] + hp[r][1] * xd[1] + hp[r][2] * xd[2]);
                lhs += ps * dotr(&hx, &nn) * a;
                d.push(x);
                n.push(nn);
                area.push(a);
                grad.push(gp);
                eg.push(grad_u(&x));
                psi.push(ps);
                gpsi.push(gs);
            }
        }
        let mut wl = Vec::new();
        let mut xw = vec![];
        for i in 0..nt {
            let (t0, t1) = (2.0 * PI * i as f64 / nt as f64, 2.0 * PI * (i + 1) as f64 / nt as f64);
            let (a, b) = ([t0.cos(), t0.sin(), 0.0], [t1.cos(), t1.sin(), 0.0]);
            let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0];
            wl.push(Wl::new(&d, &n, &[], mid, [b[0] - a[0], b[1] - a[1], 0.0]));
            xw.push(disp(&mid));
        }
        let gradw: Vec<C3> = wl.iter().map(|w| w.at(&grad)).collect();
        let x = State { w: CZ, alpha, x: d.iter().map(disp).collect(), grad: vec![], xw, gradw: vec![], eg };
        let ph = State { w: CZ, alpha, x: vec![], grad, xw: vec![], gradw, eg: vec![] };
        let free = vec![true; wl.len()];
        let mo = Motion { bi: 0, idx: vec![], d, n, area, wl, free, states: vec![], nh: 1, n6: 1, psi, gpsi };
        let rhs = mo.stokes(&x, &ph, 0, 0);
        assert!((rhs - lhs).norm() <= 2e-3 * lhs.norm(), "{lhs} {rhs}");
        // without the deformation terms the transform is wrong by O(1)
        let x0 = State { eg: vec![], ..x };
        let wrong = mo.stokes(&x0, &ph, 0, 0);
        assert!((wrong - lhs).norm() > 0.05 * lhs.norm(), "{lhs} {wrong}");
    }
}
