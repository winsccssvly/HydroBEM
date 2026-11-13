//! Mean drift force and moment of one body by the momentum flux through a control
//! surface (theory §11.3, both alternatives of the free-surface integral): the
//! submerged part S_c, its free-surface part S_f, their common line CL, the body
//! waterline WL and the hydrostatic part.
//!
//! Means of products of complex amplitudes: <a b> = Re(a conj(b)) / 2; the time factor
//! e^{i w t}, zeta = -(i w / g) phi.

use num_complex::Complex64;

use crate::geometry::Vec3;
use crate::model_codes::WaterlineTerm;
use crate::pressure_drift::about_rotation_center;

#[derive(Clone, Debug)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct ControlSurfaceDrift {
    al_: [Complex64; 3], // rotation and translation of the origin O
    xo_: [Complex64; 3],
    xr_: [Complex64; 3], // the translation of R
    rc_: Vec3,           // the rotation centre R about O (body axes), the point of the moments
    rho_: f64,
    g_: f64,
    w_: f64,
    alt_: WaterlineTerm,
    F_: [f64; 6],
}

fn mean(a: Complex64, b: Complex64) -> f64 {
    0.5 * (a * b.conj()).re
}

impl ControlSurfaceDrift {
    /// X: the six rigid motions of the body about its rotation centre Rc (body axes);
    /// alternative ControlSurface: -rho g / 2 int_CL n' zeta^2; Body: +rho g / 2 int_WL n' zeta^2
    /// minus rho g int_Sf zeta grad' zeta, equal by the divergence theorem (theory §11.3).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn new(
        X: &[Complex64; 6],
        Rc: &Vec3,
        rho: f64,
        gravity: f64,
        omega: f64,
        alternative: WaterlineTerm,
    ) -> ControlSurfaceDrift {
        let al_ = [X[3], X[4], X[5]];
        // the translation of O: xi_R - alpha x R
        let aR = [al_[1] * Rc[2] - al_[2] * Rc[1], al_[2] * Rc[0] - al_[0] * Rc[2], al_[0] * Rc[1] - al_[1] * Rc[0]];
        let xo_ = [X[0] - aR[0], X[1] - aR[1], X[2] - aR[2]];
        let xr_ = [X[0], X[1], X[2]];
        ControlSurfaceDrift {
            al_,
            xo_,
            xr_,
            rc_: *Rc,
            rho_: rho,
            g_: gravity,
            w_: omega,
            alt_: alternative,
            F_: [0.0; 6],
        }
    }

    /// A point of S_c: the total velocity v (body axes), the point xb about the body origin O,
    /// the normal n pointing away from the body (out of the fluid inside S_c), the area weight.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn submerged(&mut self, v: &[Complex64; 3], xb: &Vec3, n: &Vec3, weight: f64) {
        // -rho int [grad phi dphi/dn - n |grad phi|^2 / 2] dS and its moment (x x)
        let vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        let vv = mean(v[0], v[0]) + mean(v[1], v[1]) + mean(v[2], v[2]);
        let mut f = [0.0f64; 3];
        for d in 0..3 {
            f[d] = -self.rho_ * weight * (mean(v[d], vn) - 0.5 * n[d] * vv);
        }
        let F_ = &mut self.F_;
        F_[0] += f[0];
        F_[1] += f[1];
        F_[2] += f[2];
        F_[3] += xb[1] * f[2] - xb[2] * f[1];
        F_[4] += xb[2] * f[0] - xb[0] * f[2];
        F_[5] += xb[0] * f[1] - xb[1] * f[0];
    }
    /// A point of S_f (z = 0): the total potential phi and velocity v.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn free_surface(&mut self, phi: Complex64, v: &[Complex64; 3], xb: &Vec3, weight: f64) {
        // rho k int [zeta dphi_t/dz + |grad phi|^2 / 2] dS and (x x k)
        let iw = Complex64::new(0.0, self.w_);
        let zeta = -iw / self.g_ * phi;
        let vv = mean(v[0], v[0]) + mean(v[1], v[1]) + mean(v[2], v[2]);
        let fz = self.rho_ * weight * (mean(zeta, iw * v[2]) + 0.5 * vv);
        let F_ = &mut self.F_;
        F_[2] += fz;
        F_[3] += xb[1] * fz;
        F_[4] -= xb[0] * fz;
        if self.alt_ == WaterlineTerm::Body {
            // -rho g int_Sf <zeta grad' zeta> dS and (x x): grad' zeta = -(i w / g) (v_x, v_y)
            let gx = -iw / self.g_ * v[0];
            let gy = -iw / self.g_ * v[1];
            let f =
                [-self.rho_ * self.g_ * weight * mean(zeta, gx), -self.rho_ * self.g_ * weight * mean(zeta, gy), 0.0];
            F_[0] += f[0];
            F_[1] += f[1];
            F_[3] += -xb[2] * f[1];
            F_[4] += xb[2] * f[0];
            F_[5] += xb[0] * f[1] - xb[1] * f[0];
        }
    }
    /// An element of CL: the total potential, the horizontal unit normal away from the body.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn line(&mut self, phi: Complex64, xb: &Vec3, nh: &Vec3, dl: f64) {
        // -rho g / 2 int_CL n' <zeta^2> dl and (x x n'); alternative 2 has the body waterline instead
        if self.alt_ == WaterlineTerm::Body {
            return;
        }
        let zeta = Complex64::new(0.0, -self.w_ / self.g_) * phi;
        let s = -0.5 * self.rho_ * self.g_ * mean(zeta, zeta) * dl;
        let f = [s * nh[0], s * nh[1], 0.0];
        let F_ = &mut self.F_;
        F_[0] += f[0];
        F_[1] += f[1];
        F_[3] += -xb[2] * f[1];
        F_[4] += xb[2] * f[0];
        F_[5] += xb[0] * f[1] - xb[1] * f[0];
    }
    /// An element of the body waterline: the total potential at its panel, the panel normal n
    /// (body axes, into the fluid).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn waterline(&mut self, phi: Complex64, xb: &Vec3, n: &Vec3, dl: f64) {
        // -rho g k int_WL <zeta (xi + alpha x x) . n'> dl + rho g / 2 k int_WL nz / sqrt(1 - nz^2)
        // <(zeta - D_z)^2> dl, n' the horizontal normal into the body, nz of the normal into the
        // body (-n here)
        let (al_, xo_) = (&self.al_, &self.xo_);
        let (rho_, g_) = (self.rho_, self.g_);
        let zeta = Complex64::new(0.0, -self.w_ / g_) * phi;
        let D = [
            xo_[0] + al_[1] * xb[2] - al_[2] * xb[1],
            xo_[1] + al_[2] * xb[0] - al_[0] * xb[2],
            xo_[2] + al_[0] * xb[1] - al_[1] * xb[0],
        ];
        let h = (n[0] * n[0] + n[1] * n[1]).max(1e-24).sqrt();
        let nhx = -n[0] / h;
        let nhy = -n[1] / h;
        let nz = -n[2];
        let Dn = D[0] * nhx + D[1] * nhy;
        let sq = (1.0 - nz * nz).max(1e-12).sqrt();
        let zr = zeta - D[2];
        let fz = (-rho_ * g_ * mean(zeta, Dn) + 0.5 * rho_ * g_ * nz / sq * mean(zr, zr)) * dl;
        let F_ = &mut self.F_;
        F_[2] += fz;
        F_[3] += xb[1] * fz;
        F_[4] -= xb[0] * fz;
        if self.alt_ == WaterlineTerm::Body {
            // +rho g / 2 int_WL n' <zeta^2> dl and (x x n'), n' horizontal into the body
            let s = 0.5 * rho_ * g_ * mean(zeta, zeta) * dl;
            let f = [s * nhx, s * nhy, 0.0];
            F_[0] += f[0];
            F_[1] += f[1];
            F_[3] += -xb[2] * f[1];
            F_[4] += xb[2] * f[0];
            F_[5] += xb[0] * f[1] - xb[1] * f[0];
        }
    }
    /// The hydrostatic part (the closed forms of theory §11.2): force and moment about O.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    pub fn add(&mut self, FS: &[f64; 6]) {
        for d in 0..6 {
            self.F_[d] += FS[d];
        }
    }
    /// The force and moment about the rotation centre R, then the moment about the moving R with
    /// the first-order force F1 on the body (body axes): o[9] (pressure_drift::about_rotation_center).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn finish(&self, F1: &[Complex64; 3], o: &mut [f64; 9]) {
        about_rotation_center(&self.F_, F1, &self.rc_, &self.xr_, &self.xo_, o);
    }
}
