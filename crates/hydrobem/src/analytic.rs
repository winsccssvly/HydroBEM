//! Analytic patches (reference/inputs-meshes §2): exact maps x(u, v)
//! on [-1, 1]^2 with u x v pointing into the body.

use crate::geometry::{Vec3, v3};
use crate::mesh_desc::{AnalyticKind, Profile};
use crate::patch_geometry::bilinear_map;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticPatch {
    pub kind: AnalyticKind,
    pub profile: Profile,
    pub a: f64, // horizontal scales of x and y
    pub b: f64,
    pub theta0: f64, // azimuth at u = -1 and u = 1 (radians)
    pub theta1: f64,
    pub p: [f64; 4],   // line: rho0 z0 rho1 z1; arc: c phi0 phi1; ellipses: a0 b0 z0 z1; segment-arc: z1
    pub vt: [f64; 12], // bilinear: the four corners; segment-arc: the segment ends
}

impl Default for AnalyticPatch {
    fn default() -> Self {
        AnalyticPatch {
            kind: AnalyticKind::Line,
            profile: Profile::Linear,
            a: 1.0,
            b: 1.0,
            theta0: 0.0,
            theta1: 0.0,
            p: [0.0; 4],
            vt: [0.0; 12],
        }
    }
}

impl AnalyticPatch {
    /// The profile P(t) of t = (v + 1) / 2 and dP/dt.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn profile_of(&self, t: f64, P: &mut f64, dP: &mut f64) {
        match self.profile {
            Profile::Smooth => {
                *P = t * t * (3.0 - 2.0 * t);
                *dP = 6.0 * t * (1.0 - t);
            }
            Profile::QuadraticStart => {
                *P = t * t;
                *dP = 2.0 * t;
            }
            Profile::QuadraticEnd => {
                *P = t * (2.0 - t);
                *dP = 2.0 * (1.0 - t);
            }
            _ => {
                *P = t;
                *dP = 1.0;
            }
        }
    }

    /// Point and derivatives with respect to u and v.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn eval(&self, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
        let vt = &self.vt;
        let p = &self.p;
        let (a, b) = (self.a, self.b);
        if self.kind == AnalyticKind::Bilinear {
            let c = |i: usize| v3(vt[3 * i], vt[3 * i + 1], vt[3 * i + 2]);
            bilinear_map(&[c(0), c(1), c(2), c(3)], u, v, x, xu, xv);
            return;
        }
        let th = self.theta0 + 0.5 * (u + 1.0) * (self.theta1 - self.theta0);
        let dth = 0.5 * (self.theta1 - self.theta0);
        let (c, s, t) = (th.cos(), th.sin(), 0.5 * (v + 1.0));
        let (mut P, mut dP) = (0.0, 0.0);
        self.profile_of(t, &mut P, &mut dP);
        if self.kind == AnalyticKind::Ellipses {
            // between two ellipses: the scales and z linear in P
            let A = p[0] + (a - p[0]) * P;
            let B = p[1] + (b - p[1]) * P;
            let z = p[2] + (p[3] - p[2]) * P;
            *x = v3(A * c, B * s, z);
            *xu = v3(-A * s * dth, B * c * dth, 0.0);
            *xv = v3(0.5 * (a - p[0]) * dP * c, 0.5 * (b - p[1]) * dP * s, 0.5 * (p[3] - p[2]) * dP);
            return;
        }
        if self.kind == AnalyticKind::SegmentArc {
            // from a segment (u from its first to its second end) to an elliptic arc
            let sg = 0.5 * (u + 1.0);
            let mut q0 = Vec3::ZERO;
            let q1 = v3(a * c, b * s, p[0]);
            for k in 0..3 {
                q0[k] = (1.0 - sg) * vt[k] + sg * vt[3 + k];
            }
            let q1u = v3(-a * s * dth, b * c * dth, 0.0);
            for k in 0..3 {
                x[k] = (1.0 - P) * q0[k] + P * q1[k];
                xu[k] = (1.0 - P) * 0.5 * (vt[3 + k] - vt[k]) + P * q1u[k];
                xv[k] = 0.5 * dP * (q1[k] - q0[k]);
            }
            return;
        }
        // rho, z and their derivatives with respect to t
        let (rho, drho, z, dz);
        if self.kind == AnalyticKind::Arc {
            let phi = p[1] + (p[2] - p[1]) * P;
            let dphi = (p[2] - p[1]) * dP;
            rho = phi.cos();
            drho = -phi.sin() * dphi;
            z = -p[0] * phi.sin();
            dz = -p[0] * phi.cos() * dphi;
        } else {
            rho = p[0] + (p[2] - p[0]) * P;
            drho = (p[2] - p[0]) * dP;
            z = p[1] + (p[3] - p[1]) * P;
            dz = (p[3] - p[1]) * dP;
        }
        *x = v3(a * rho * c, b * rho * s, z);
        *xu = v3(-a * rho * s * dth, b * rho * c * dth, 0.0);
        *xv = v3(0.5 * a * drho * c, 0.5 * b * drho * s, 0.5 * dz); // dt/dv = 1/2
    }
}
