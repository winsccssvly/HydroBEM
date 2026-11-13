//! The hydrostatics of a body (theory §3.2: volumes, centre of buoyancy, waterplane, restoring
//! matrix with the internal tanks of theory §10.2) and its rigid inertia about the rotation
//! centre (theory §3.1), with the symmetric eigen-decomposition the inertia rule uses.

use crate::geometry::{Vec3, dot, reflections};
use crate::model_codes::{Mass, Origin};
use crate::problem::{Body, Problem, Role};
use crate::quadrature::{QPoint, record_quadrature};

#[derive(Clone, Copy, Debug)]
pub struct Hydrostatics {
    pub volumes: [f64; 3], // the three estimates of the divergence theorem (x n_x, y n_y, z n_z)
    pub volume: f64,       // their mean
    pub cb: [f64; 3],      // body coordinates
    pub wp_area: f64,
    pub wp_centroid: [f64; 2], // x, y in body coordinates
    pub c_rho_g: [f64; 36],
    pub has_weight: bool, // a centre of gravity and a mass: the weight terms are in c
    pub mass: f64,        // NaN without a mass
    pub cg: [f64; 3],     // body coordinates (the input); NaN without a centre of gravity
    pub c: [f64; 36],
    pub displaced_mass: f64, // rho V with V = volumes[0], the volume of the displaced mass
    pub metacentric_height: [f64; 2], // C44 / (rho g V), C55 / (rho g V) with the weight terms; NaN without
}

impl Default for Hydrostatics {
    fn default() -> Self {
        Hydrostatics {
            volumes: [0.0; 3],
            volume: 0.0,
            cb: [0.0; 3],
            wp_area: 0.0,
            wp_centroid: [0.0; 2],
            c_rho_g: [0.0; 36],
            has_weight: false,
            mass: 0.0,
            cg: [0.0; 3],
            c: [0.0; 36],
            displaced_mass: 0.0,
            metacentric_height: [f64::NAN; 2],
        }
    }
}

// 3x3 helpers for the rigid inertia (theory §3.1)
type M3 = [f64; 9];
fn skew(v: &Vec3) -> M3 {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}

/// The eigenvalues and eigenvectors of a symmetric n x n matrix (row-major) by cyclic Jacobi
/// rotations (Golub & Van Loan 2013, §8.5): (values, vectors), vector k in the column k of the
/// row-major n x n `vectors`. For the small inertia blocks of rule 10.
pub(crate) fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        let all: f64 = a.iter().map(|x| x * x).sum();
        if off <= 1e-30 * all || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

// the sums of the hydrostatics over the wetted surface (theory §3.2)
#[derive(Default)]
pub(crate) struct Acc {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub mx: f64,
    pub my: f64,
    pub mz: f64,
    pub aw: f64,
    pub ax: f64,
    pub ay: f64,
    pub axx: f64,
    pub ayy: f64,
    pub axy: f64,
}

impl Acc {
    fn accumulate(&mut self, b: &Body, q: &QPoint, waterplane: bool) {
        let x = q.x[0] - b.rotation_center[0];
        let y = q.x[1] - b.rotation_center[1];
        let z = q.x[2] + b.position[2];
        self.add(x, y, z, q.nds, waterplane);
    }
    /// The volume, its moments and (waterplane) the waterplane moments of a quadrature point at
    /// (x, y, z) with nds (theory §3.2).
    pub fn add(&mut self, x: f64, y: f64, z: f64, nds: Vec3, waterplane: bool) {
        self.vx += x * nds[0];
        self.vy += y * nds[1];
        self.vz += z * nds[2];
        self.mx += 0.5 * x * x * nds[0];
        self.my += 0.5 * y * y * nds[1];
        self.mz += 0.5 * z * z * nds[2];
        if waterplane {
            self.aw -= nds[2];
            self.ax -= x * nds[2];
            self.ay -= y * nds[2];
            self.axx -= x * x * nds[2];
            self.ayy -= y * y * nds[2];
            self.axy -= x * y * nds[2];
        }
    }
}

impl Problem {
    /// Hydrostatics of a body (theory §3.2): body axes, the stiffness about R, the centres of buoyancy
    /// and gravity in body coordinates. body: zero-based body index. Returns volumes, buoyancy and
    /// water-plane data, stiffness; mass = Body::mass or the displaced mass rho V_x.
    // Hydrostatics of theory §3.2: body axes, horizontal coordinates relative to R,
    // vertical coordinate measured from the free surface.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn hydrostatics(&self, body: i32) -> Hydrostatics {
        let bi = body as usize;
        let b = &self.bodies[bi];
        let m = &self.meshes[b.mesh as usize];
        let d = &self.derived_[bi];
        let env = &self.env;
        let zr = b.rotation_center[2] + b.position[2]; // global z of R
        let mut s = Acc::default();
        for r in reflections(m.isx, m.isy) {
            for rec in 0..m.count {
                let role = d.role[rec as usize];
                if role != Role::Wet && role != Role::Pressure && role != Role::Plate {
                    continue;
                }
                // a free-surface plate (normal -z, in z = 0) adds no volume, its area to the waterplane
                for q in record_quadrature(m, rec, &r) {
                    s.accumulate(b, &q, role != Role::Pressure);
                }
            }
        }
        for p in &b.plates {
            if let Some(pi) = p.mesh {
                let pm = &self.meshes[pi];
                for r in reflections(pm.isx, pm.isy) {
                    for rec in 0..pm.count {
                        for q in record_quadrature(pm, rec, &r) {
                            s.accumulate(b, &q, true);
                        }
                    }
                }
            }
        }
        for p in &b.pressure {
            if let Some(pi) = p.mesh {
                let pm = &self.meshes[pi];
                for r in reflections(pm.isx, pm.isy) {
                    for rec in 0..pm.count {
                        for q in record_quadrature(pm, rec, &r) {
                            s.accumulate(b, &q, false);
                        }
                    }
                }
            }
        }
        let V = s.vz;
        // the centre of buoyancy relative to R, then in body coordinates
        let cb = [s.mx / V, s.my / V, s.mz / V - zr];
        let mut h = Hydrostatics {
            volumes: [s.vx, s.vy, s.vz],
            volume: (s.vx + s.vy + s.vz) / 3.0, // as the mesh geometry's
            cb: std::array::from_fn(|i| cb[i] + b.rotation_center[i]),
            ..Hydrostatics::default()
        };
        h.wp_area = s.aw;
        // the waterplane centroid relative to R, then in body coordinates (as the centres)
        h.wp_centroid[0] = s.ax / s.aw + b.rotation_center[0];
        h.wp_centroid[1] = s.ay / s.aw + b.rotation_center[1];
        let rg = env.density * env.gravity;
        let Vzb = s.mz - zr * V; // V z_b relative to R
        {
            let c = &mut h.c_rho_g;
            c[2 * 6 + 2] = rg * s.aw;
            c[2 * 6 + 3] = rg * s.ay;
            c[3 * 6 + 2] = c[2 * 6 + 3];
            c[2 * 6 + 4] = -rg * s.ax;
            c[4 * 6 + 2] = c[2 * 6 + 4];
            c[3 * 6 + 3] = rg * (s.ayy + Vzb);
            c[3 * 6 + 4] = -rg * s.axy;
            c[4 * 6 + 3] = c[3 * 6 + 4];
            c[3 * 6 + 5] = -rg * s.mx;
            c[4 * 6 + 4] = rg * (s.axx + Vzb);
            c[4 * 6 + 5] = -rg * s.my;
        }
        // internal tanks (theory §10.2): the same formulas over the tank walls, the
        // normals into the tank fluid, z from the tank free surface, the tank density
        let mut tank_mass = 0.0;
        for ti in 0..b.tanks.len() {
            let zt = d.tank_z[ti] + b.position[2];
            let rgt = rg * b.tanks[ti].density;
            let (mut tv, mut tmx, mut tmy, mut tmz, mut ta) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut tax, mut tay, mut taxx, mut tayy, mut taxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            // the walls and the plates of the tank's free surface (normal -z into the tank fluid, at
            // z = 0 from its surface: no volume, their area and moments leave its free-surface terms)
            let mut quad = Vec::new();
            for r in reflections(m.isx, m.isy) {
                for rec in 0..m.count {
                    if d.tank[rec as usize] == ti as i32 && d.role[rec as usize] == Role::Tank {
                        quad.extend(record_quadrature(m, rec, &r));
                    }
                }
            }
            for pi in b.tanks[ti].plates.iter().filter_map(|p| p.mesh) {
                let pm = &self.meshes[pi];
                for r in reflections(pm.isx, pm.isy) {
                    for rec in 0..pm.count {
                        quad.extend(record_quadrature(pm, rec, &r));
                    }
                }
            }
            for q in quad {
                let x = q.x[0] - b.rotation_center[0];
                let y = q.x[1] - b.rotation_center[1];
                let z = q.x[2] + b.position[2] - zt;
                tv += z * q.nds[2];
                tmx += 0.5 * x * x * q.nds[0];
                tmy += 0.5 * y * y * q.nds[1];
                tmz += 0.5 * z * z * q.nds[2];
                ta -= q.nds[2];
                tax -= x * q.nds[2];
                tay -= y * q.nds[2];
                taxx -= x * x * q.nds[2];
                tayy -= y * y * q.nds[2];
                taxy -= x * y * q.nds[2];
            }
            let tvz = tmz - (zr - zt) * tv; // as V z_b about R
            let c = &mut h.c_rho_g;
            c[2 * 6 + 2] += rgt * ta;
            c[2 * 6 + 3] += rgt * tay;
            c[3 * 6 + 2] += rgt * tay;
            c[2 * 6 + 4] -= rgt * tax;
            c[4 * 6 + 2] -= rgt * tax;
            c[3 * 6 + 3] += rgt * (tayy + tvz);
            c[3 * 6 + 4] -= rgt * taxy;
            c[4 * 6 + 3] -= rgt * taxy;
            c[3 * 6 + 5] -= rgt * tmx;
            c[4 * 6 + 4] += rgt * (taxx + tvz);
            c[4 * 6 + 5] -= rgt * tmy;
            tank_mass += env.density * b.tanks[ti].density * (-tv); // the tank volume is -tv (normals into the tank)
        }
        h.c = h.c_rho_g;
        // the displaced mass less the tank fluid; no mass given: none in use; the weight needs the mass
        // and the centre of gravity
        h.mass = match b.mass {
            None => f64::NAN,
            Some(Mass::Given(mass)) => mass,
            Some(Mass::Displaced) => env.density * s.vx - tank_mass,
        };
        h.has_weight = b.has_cg && b.mass.is_some();
        let g = b.cg - b.rotation_center;
        for i in 0..3 {
            h.cg[i] = if b.has_cg { b.cg[i] } else { f64::NAN }; // given or not, as the mass
        }
        if h.has_weight {
            let mg = h.mass * env.gravity;
            h.c[3 * 6 + 3] -= mg * g[2];
            h.c[3 * 6 + 5] += mg * g[0];
            h.c[4 * 6 + 4] -= mg * g[2];
            h.c[4 * 6 + 5] += mg * g[1];
        }
        // the displaced mass and the metacentric heights GM = C / (rho g V) with the weight terms
        // (none without a volume: a body of sheets or plates only)
        h.displaced_mass = env.density * s.vx;
        if h.has_weight && s.vx != 0.0 {
            let w = rg * s.vx;
            h.metacentric_height = [h.c[3 * 6 + 3] / w, h.c[4 * 6 + 4] / w];
        }
        h
    }
}

/// Rigid inertia about R (theory §3.1), 6 x 6 row-major.
/// b: body with a centre of gravity and inertia (components, or radii of gyration:
/// I_ij = m r_ij |r_ij|), moved to R; m: the mass in use (Hydrostatics::mass).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn rigid_inertia(b: &Body, m: f64) -> [f64; 36] {
    let c = b.cg - b.rotation_center;
    let mut J0: M3 = [
        b.inertia[0],
        b.inertia[1],
        b.inertia[2],
        b.inertia[1],
        b.inertia[3],
        b.inertia[4],
        b.inertia[2],
        b.inertia[4],
        b.inertia[5],
    ];
    if b.has_gyration {
        // I_ij = m r_ij |r_ij| with the mass in use
        for k in 0..9 {
            J0[k] = m * b.gyration[k] * b.gyration[k].abs();
        }
    }
    let mut JR = J0;
    if b.inertia_origin != Origin::RotationCenter {
        let o = if b.inertia_origin == Origin::CenterOfGravity { b.cg } else { b.inertia_point };
        let a = b.cg - o;
        for i in 0..3 {
            for j in 0..3 {
                JR[3 * i + j] += -m * ((if i == j { dot(a, a) } else { 0.0 }) - a[i] * a[j])
                    + m * ((if i == j { dot(c, c) } else { 0.0 }) - c[i] * c[j]);
            }
        }
    }
    let mut M = [0.0f64; 36];
    let S = skew(&c);
    for i in 0..3 {
        M[i * 6 + i] = m;
        for j in 0..3 {
            M[i * 6 + 3 + j] = -m * S[3 * i + j];
            M[(3 + i) * 6 + j] = m * S[3 * i + j];
            M[(3 + i) * 6 + 3 + j] = JR[3 * i + j];
        }
    }
    M
}
