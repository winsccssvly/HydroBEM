//! Mean drift force and moment of one body by pressure integration (theory §11.2),
//! shared by both methods: the pressure expanded about the mean wetted surface, the
//! waterline strip and, with pressure surfaces, the hydrostatic part by its closed forms
//! (equal to the surface integral by the divergence theorem). With generalized modes (theory §10.1)
//! their displacement joins that of the rigid motions, their displacement gradient turns and
//! stretches the area element (Nanson's formula), and their own rows are the generalized forces
//! (the virtual work of the pressure on the exact surface against the mode shape).

use std::ops::{Add, Mul, Neg, Sub};

use num_complex::Complex64;

use crate::geometry::{Vec3, cross, v3};

/// A complex vector of three components.
pub type Vec3c = [Complex64; 3];

#[derive(Clone, Debug)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct PressureDrift {
    al_: Vec3c, // rotation and translation of the origin O
    xo_: Vec3c,
    rc_: Vec3,         // the rotation centre R about O (body axes), the point of the moments
    H_: [[f64; 3]; 3], // second-order rotation (lower triangular), applied about R
    HR_: Vec3,         // H R (zero for R = 0)
    X_: [Complex64; 6],
    rho_: f64,
    g_: f64,
    w_: f64,
    cf_: bool,
    F_: [f64; 6],
    F1_: Vec3c, // the first-order force, for the moment about the moving rotation centre
    cs_: ClosedForms,
    xg_: Vec<Complex64>, // the motions of the generalized modes of the body
    Fg_: Vec<f64>,       // their generalized forces
}

/// The generalized modes at a point of the mean wetted surface (body axes): the displacement S_g
/// and its gradient B_g (B[i][k] = ∂S_i/∂x_k) of each mode.
pub struct ModeShapes<'a> {
    pub s: &'a [Vec3],
    pub b: &'a [[[f64; 3]; 3]],
}

/// Two unit tangents t1, t2 of the unit normal n with t1 × t2 = n.
pub fn tangents(n: &Vec3) -> (Vec3, Vec3) {
    let a = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        v3(1.0, 0.0, 0.0)
    } else if n[1].abs() <= n[2].abs() {
        v3(0.0, 1.0, 0.0)
    } else {
        v3(0.0, 0.0, 1.0)
    };
    let t = cross(*n, a);
    let l = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let t1 = v3(t[0] / l, t[1] / l, t[2] / l);
    (t1, cross(*n, t1))
}

/// E v for a complex 3 x 3 matrix and a real vector.
pub fn mat_vec(e: &[[Complex64; 3]; 3], v: &Vec3) -> Vec3c {
    std::array::from_fn(|i| e[i][0] * v[0] + e[i][1] * v[1] + e[i][2] * v[2])
}

/// The first-order change of the area-weighted normal by a displacement gradient E (Nanson's
/// formula to first order): (tr E) n − Eᵀ n.
pub fn nanson(e: &[[Complex64; 3]; 3], n: &Vec3) -> Vec3c {
    let tr = e[0][0] + e[1][1] + e[2][2];
    std::array::from_fn(|k| tr * n[k] - (e[0][k] * n[0] + e[1][k] * n[1] + e[2][k] * n[2]))
}

fn cross_c(a: &Vec3c, b: &Vec3c) -> Vec3c {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

// <a b> for complex amplitudes: Re(a conj(b)) / 2
fn mean(a: Complex64, b: Complex64) -> f64 {
    0.5 * (a * b.conj()).re
}

// <a × b> for complex amplitudes: Re(a × conj(b)) / 2
fn mean_cross(a: &Vec3c, b: &Vec3c) -> Vec3 {
    let c = cross_c(a, &[b[0].conj(), b[1].conj(), b[2].conj()]);
    v3(0.5 * c[0].re, 0.5 * c[1].re, 0.5 * c[2].re)
}

// x × a of a real and a complex vector
fn cross_rc(x: &Vec3, a: &Vec3c) -> Vec3c {
    [x[1] * a[2] - x[2] * a[1], x[2] * a[0] - x[0] * a[2], x[0] * a[1] - x[1] * a[0]]
}

fn complex3(v: &Vec3) -> Vec3c {
    [Complex64::new(v[0], 0.0), Complex64::new(v[1], 0.0), Complex64::new(v[2], 0.0)]
}

/// The mean force and moment F (about O, body axes) of a body as nine values about its rotation
/// centre R (rc, about O): the force, the moment about R, M_R = M_O − R × F, and the moment about
/// the moving R, M_R − <ξ_R × F1> with ξ_R (xr) the translation of R and F1 the first-order force.
/// R is body-fixed and the moments are in the body's mean axes, so the transfer is exact; the
/// moment about the moving R differs from the transfer of that about the moving O (xo: the
/// translation of O) by −<(α × R) × F1>. For R = 0 the arithmetic of the moments about O.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn about_rotation_center(F: &[f64; 6], F1: &Vec3c, rc: &Vec3, xr: &Vec3c, xo: &Vec3c, o: &mut [f64; 9]) {
    o[..6].copy_from_slice(F);
    let at_o = *rc == Vec3::ZERO;
    if !at_o {
        let rf = cross(*rc, v3(F[0], F[1], F[2]));
        for d in 0..3 {
            o[3 + d] = F[3 + d] - rf[d];
        }
    }
    let xi = if at_o { xo } else { xr };
    let F1c = [F1[0].conj(), F1[1].conj(), F1[2].conj()];
    let xF = cross_c(xi, &F1c);
    for d in 0..3 {
        o[6 + d] = o[3 + d] - 0.5 * xF[d].re;
    }
}

/// The sums of the hydrostatic closed forms of theory §11.2 about O in body axes: A_wp, x_f, y_f
/// and L_ij of the wetted surface, the volume moments of the wetted and pressure surfaces; one
/// implementation for the drift (the means of the motions) and the QTFs (their pair products).
#[derive(Clone, Copy, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct ClosedForms {
    Awp: f64,
    Axf: f64,
    Ayf: f64,
    L11: f64,
    L22: f64,
    L12: f64,
    Vxb: f64,
    Vyb: f64,
    Vb: f64,
}

impl ClosedForms {
    /// A point of the wetted (wetted = true) or pressure surface: x about O and n in body axes,
    /// its global z and weight.
    pub fn add(&mut self, x: &Vec3, n: &Vec3, z: f64, weight: f64, wetted: bool) {
        let zn = weight * z * n[2];
        self.Vxb += x[0] * zn;
        self.Vyb += x[1] * zn;
        self.Vb += zn;
        if !wetted {
            return;
        }
        self.Awp -= weight * n[2];
        self.Axf -= weight * x[0] * n[2];
        self.Ayf -= weight * x[1] * n[2];
        self.L11 -= weight * x[0] * x[0] * n[2];
        self.L22 -= weight * x[1] * x[1] * n[2];
        self.L12 -= weight * x[0] * x[1] * n[2];
    }

    /// The hydrostatic parts of F_z, M_x and M_y over rho g (F[2] less, F[3] and F[4] more) from the
    /// products M(i, k) of the motions of O (0..2 the translation, 3..5 the rotation: their means
    /// for the drift, their pair products for the QTFs); zo the global z of O.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn terms<T>(&self, zo: f64, M: impl Fn(usize, usize) -> T) -> [T; 3]
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Neg<Output = T> + Mul<f64, Output = T>,
        f64: Mul<T, Output = T>,
    {
        let ClosedForms { Awp, Axf, Ayf, L11, L22, L12, Vxb, Vyb, .. } = *self;
        [
            M(3, 5) * Axf + M(4, 5) * Ayf + 0.5 * (M(3, 3) + M(4, 4)) * Awp * zo,
            (M(1, 4) - M(2, 5)) * Axf - M(1, 3) * Ayf + M(3, 4) * Axf * zo
                - (1.5 * M(3, 3) + 0.5 * M(4, 4)) * Ayf * zo
                - M(1, 2) * Awp
                - M(2, 3) * Awp * zo
                + M(4, 5) * (L11 - L22)
                - 2.0 * M(3, 5) * L12
                + M(3, 4) * Vxb
                - 0.5 * (M(3, 3) + M(5, 5)) * Vyb,
            -M(0, 4) * Axf + (M(0, 3) - M(2, 5)) * Ayf + (0.5 * M(3, 3) + 1.5 * M(4, 4)) * Axf * zo
                - M(3, 4) * Ayf * zo
                + M(0, 2) * Awp
                - M(2, 4) * Awp * zo
                + M(3, 5) * (L11 - L22)
                + 2.0 * M(4, 5) * L12
                + 0.5 * (M(4, 4) + M(5, 5)) * Vxb,
        ]
    }

    /// The hydrostatic parts of F_z, M_x and M_y over rho g, as `terms`, of a rigid second-order
    /// translation d of the body (the second-order rotation H applied about R moves O by
    /// d = −H R, theory §11.2): −∫ P n and −∫ P (x × n) with the pressure −ρg d_z and the turn
    /// d × n of the moment arm, ρg d_z ∫ n dS = −ρg A_wp d_z e_z and
    /// ρg [d_z ∫ x × n dS + d × ∫ z n dS] = ρg [d_z (−A y_f, A x_f, 0) + V (d_y, −d_x, 0)] (the
    /// divergence theorem over the body closed by the water plane; V of the points added).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn translation<T>(&self, d: [T; 3]) -> [T; 3]
    where
        T: Copy + Sub<Output = T> + Mul<f64, Output = T>,
    {
        let ClosedForms { Awp, Axf, Ayf, Vb, .. } = *self;
        [d[2] * Awp, d[1] * Vb - d[2] * Ayf, d[2] * Axf - d[0] * Vb]
    }
}

impl PressureDrift {
    /// X: the six rigid motions of the body about its rotation centre Rc (body axes);
    /// closed_forms: the hydrostatic part by the closed forms (a body with pressure surfaces).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn new(X: &[Complex64; 6], Rc: &Vec3, rho: f64, gravity: f64, omega: f64, closed_forms: bool) -> PressureDrift {
        let al_ = [X[3], X[4], X[5]];
        let aR = cross_c(&al_, &complex3(Rc));
        let xo_ = [X[0] - aR[0], X[1] - aR[1], X[2] - aR[2]];
        let mut aa = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                aa[i][k] = mean(al_[i], al_[k]);
            }
        }
        let H = [
            [-0.5 * (aa[1][1] + aa[2][2]), 0.0, 0.0],
            [aa[0][1], -0.5 * (aa[0][0] + aa[2][2]), 0.0],
            [aa[0][2], aa[1][2], -0.5 * (aa[0][0] + aa[1][1])],
        ];
        let z = Complex64::new(0.0, 0.0);
        let HR = |r: &Vec3| {
            v3(H[0][0] * r[0], H[1][0] * r[0] + H[1][1] * r[1], H[2][0] * r[0] + H[2][1] * r[1] + H[2][2] * r[2])
        };
        PressureDrift {
            al_,
            xo_,
            rc_: *Rc,
            H_: H,
            HR_: if *Rc == Vec3::ZERO { Vec3::ZERO } else { HR(Rc) },
            X_: *X,
            rho_: rho,
            g_: gravity,
            w_: omega,
            cf_: closed_forms,
            F_: [0.0; 6],
            F1_: [z; 3],
            cs_: ClosedForms::default(),
            xg_: Vec::new(),
            Fg_: Vec::new(),
        }
    }

    /// The motions of the body's generalized modes (before the points; none for a rigid body).
    pub fn generalized(&mut self, xg: &[Complex64]) {
        self.xg_ = xg.to_vec();
        self.Fg_ = vec![0.0; xg.len()];
    }

    /// The generalized forces of the body's generalized modes (after the points).
    pub fn generalized_forces(&self) -> &[f64] {
        &self.Fg_
    }

    /// A point of the mean wetted surface: total potential phi and velocity v (body axes), the
    /// point xb about the body origin O and the normal n (body axes, into the fluid), its global
    /// z and its quadrature weight.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn surface(&mut self, phi: Complex64, v: &[Complex64; 3], xb: &Vec3, n: &Vec3, z: f64, weight: f64) {
        self.surface_with(phi, v, xb, n, z, weight, None);
    }

    /// A point of the mean wetted surface as `surface`, with the shapes of the body's generalized
    /// modes there (theory §10.1: their displacement in X, their gradient in the normal, their rows).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    pub fn surface_with(
        &mut self,
        phi: Complex64,
        v: &[Complex64; 3],
        xb: &Vec3,
        n: &Vec3,
        z: f64,
        weight: f64,
        shapes: Option<&ModeShapes>,
    ) {
        let iw = Complex64::new(0.0, self.w_);
        let ax = cross_c(&self.al_, &complex3(xb));
        let mut Dp = [self.xo_[0] + ax[0], self.xo_[1] + ax[1], self.xo_[2] + ax[2]]; // displacement of the point
        // the generalized modes: their displacement Xg = Σ ξ_g S_g and gradient Eg = Σ ξ_g B_g
        let gmodes = shapes.filter(|_| !self.xg_.is_empty()).map(|sh| {
            let z = Complex64::new(0.0, 0.0);
            let mut xg = [z; 3];
            let mut eg = [[z; 3]; 3];
            for (k, x) in self.xg_.iter().enumerate() {
                for i in 0..3 {
                    xg[i] += x * sh.s[k][i];
                    for l in 0..3 {
                        eg[i][l] += x * sh.b[k][i][l];
                    }
                }
            }
            (xg, eg)
        });
        if let Some((xg, _)) = &gmodes {
            for d in 0..3 {
                Dp[d] += xg[d];
            }
        }
        // the second-order rotation about R, H (x − R): the second-order translation of R zero
        let at_o = self.rc_ == Vec3::ZERO;
        let Hx = if at_o { self.Hv(xb) } else { self.Hv(&(*xb - self.rc_)) };
        let mut p2 = 0.0;
        for d in 0..3 {
            p2 += 0.5 * mean(v[d], v[d]) + mean(Dp[d], iw * v[d]);
        }
        let p2 = -self.rho_ * (p2 + if self.cf_ { 0.0 } else { self.g_ * Hx[2] });
        let p1 = -self.rho_ * (iw * phi + if self.cf_ { Complex64::new(0.0, 0.0) } else { self.g_ * Dp[2] });
        let p0 = if self.cf_ { 0.0 } else { -self.rho_ * self.g_ * z };
        let nc = complex3(n);
        let an = cross_c(&self.al_, &nc);
        let xn = cross(*xb, *n);
        let axn = cross_c(&self.al_, &complex3(&xn));
        let xin = cross_c(&self.xo_, &nc);
        let anc = [an[0].conj(), an[1].conj(), an[2].conj()];
        let xa = cross_c(&self.xo_, &anc); // <xi x (alpha x n)> = Re(xi x conj(alpha x n)) / 2
        let Hn = self.Hv(n);
        // the second-order part of X̂ × N̂, H(x − R) × n + x × H n + <(ξ + α × x) × (α × n)>, is
        // H (x × n) − (H R) × n + <ξ × (α × n)>
        let mut Hxn = self.Hv(&xn);
        if !at_o {
            Hxn -= cross(self.HR_, *n);
        }
        for d in 0..3 {
            self.F1_[d] -= weight * (p1 * n[d] + p0 * an[d]);
            self.F_[d] -= weight * (p2 * n[d] + mean(p1, an[d]) + p0 * Hn[d]);
            self.F_[3 + d] -= weight * (p2 * xn[d] + mean(p1, axn[d] + xin[d]) + p0 * (Hxn[d] + 0.5 * xa[d].re));
        }
        let (Some((xg, eg)), Some(sh)) = (gmodes, shapes) else { return };
        // the area-weighted normal N̂ dŜ = cof(F) n dS of the exact surface, F = I + A + H + Eg:
        // to first order the rigid α × n and (tr Eg) n − Egᵀ n; to second order the rigid H n and
        // the means of the cross products of the turned tangents with Eg, <Eg t1 × Eg t2> +
        // <A t1 × Eg t2> + <Eg t1 × A t2> (A t = α × t)
        let n1g = nanson(&eg, n);
        let (t1, t2) = tangents(n);
        let (e1, e2) = (mat_vec(&eg, &t1), mat_vec(&eg, &t2));
        let (a1, a2) = (cross_c(&self.al_, &complex3(&t1)), cross_c(&self.al_, &complex3(&t2)));
        let (c1, c2, c3) = (mean_cross(&e1, &e2), mean_cross(&a1, &e2), mean_cross(&e1, &a2));
        let n2g = v3(c1[0] + c2[0] + c3[0], c1[1] + c2[1] + c3[1], c1[2] + c2[2] + c3[2]);
        // the moment, X̂ × N̂ to second order: its generalized parts x × N1g + Xg × n with P1 and
        // x × N2g + <Xg × N1> + <Xr × N1g> with P0 (Xr = ξ + α × x, N1 = α × n + N1g)
        let xr = [self.xo_[0] + ax[0], self.xo_[1] + ax[1], self.xo_[2] + ax[2]];
        let n1 = [an[0] + n1g[0], an[1] + n1g[1], an[2] + n1g[2]];
        let xn1g = cross_rc(xb, &n1g);
        let xgn = cross_c(&xg, &nc);
        let xn2g = cross(*xb, n2g);
        let (m1, m2) = (mean_cross(&xg, &n1), mean_cross(&xr, &n1g));
        for d in 0..3 {
            self.F1_[d] -= weight * p0 * n1g[d];
            self.F_[d] -= weight * (mean(p1, n1g[d]) + p0 * n2g[d]);
            self.F_[3 + d] -= weight * (mean(p1, xn1g[d] + xgn[d]) + p0 * (xn2g[d] + m1[d] + m2[d]));
        }
        // the generalized forces: −∫ P N̂·S_g dŜ, S_g at the material point
        let n2 = v3(Hn[0] + n2g[0], Hn[1] + n2g[1], Hn[2] + n2g[2]);
        for (k, s) in sh.s.iter().enumerate().take(self.Fg_.len()) {
            let sn = s[0] * n[0] + s[1] * n[1] + s[2] * n[2];
            let sn1 = s[0] * n1[0] + s[1] * n1[1] + s[2] * n1[2];
            let sn2 = s[0] * n2[0] + s[1] * n2[1] + s[2] * n2[2];
            self.Fg_[k] -= weight * (p2 * sn + mean(p1, sn1) + p0 * sn2);
        }
    }
    /// A waterline element of length dl at xb (about O) with the normal n of its panel (body axes).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn waterline(&mut self, phi: Complex64, xb: &Vec3, n: &Vec3, dl: f64) {
        self.waterline_with(phi, xb, n, dl, None);
    }

    /// A waterline element as `waterline`, with the displacements of the body's generalized modes
    /// there.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn waterline_with(&mut self, phi: Complex64, xb: &Vec3, n: &Vec3, dl: f64, shapes: Option<&[Vec3]>) {
        // -rho g / 2 <zeta_r^2> n / sqrt(1 - nz^2) per unit length (zeta_r: the relative elevation)
        let zeta = Complex64::new(0.0, -self.w_ / self.g_) * phi;
        let D = self.vertical(xb, shapes);
        let shapes = shapes.filter(|_| !self.xg_.is_empty());
        let zr = zeta - D;
        let sq = (1.0 - n[2] * n[2]).max(1e-12).sqrt();
        let sw = -0.5 * self.rho_ * self.g_ * mean(zr, zr) * dl / sq;
        // closed forms: the horizontal part without its hydrostatic term <D^2> (theory §11.2)
        let swh =
            if self.cf_ { -0.5 * self.rho_ * self.g_ * (mean(zeta, zeta) - 2.0 * mean(zeta, D)) * dl / sq } else { sw };
        let nh = v3(n[0], n[1], 0.0);
        let nv = v3(0.0, 0.0, n[2]);
        let xh = cross(*xb, nh);
        let xv = cross(*xb, nv);
        for d in 0..3 {
            self.F_[d] += swh * nh[d] + sw * nv[d];
            self.F_[3 + d] += swh * xh[d] + sw * xv[d];
        }
        if let Some(s) = shapes {
            // (never with the closed forms: a body with generalized modes has no pressure surfaces)
            for (k, f) in self.Fg_.iter_mut().enumerate() {
                *f += sw * (s[k][0] * n[0] + s[k][1] * n[1] + s[k][2] * n[2]);
            }
        }
    }
    /// The vertical first-order displacement X_3 of the body at the point xb (about O, body axes),
    /// with the displacements of its generalized modes there (theory (11.10)).
    pub fn vertical(&self, xb: &Vec3, shapes: Option<&[Vec3]>) -> Complex64 {
        let ax = cross_c(&self.al_, &complex3(xb));
        let mut d = self.xo_[2] + ax[2];
        if let Some(s) = shapes.filter(|_| !self.xg_.is_empty()) {
            for (k, x) in self.xg_.iter().enumerate() {
                d += x * s[k][2];
            }
        }
        d
    }

    /// A waterline element of a wall under a free-surface plate whose edge lies on the wall
    /// (theory §11.2): the band between the wall's displaced waterline X_3 and the plate's edge at
    /// the vertical displacement `top` (of the plate's body and modes there), with the pressure
    /// P1_w − ρg s at the height s above the wall's waterline, P1_w = −ρ(iωφ + g X_3) of the wall's
    /// potential φ there. Its force per unit length is −<P1_w ζ_r − ½ρg ζ_r²> n / √(1 − n_z²) with
    /// ζ_r = top − X_3: under a free surface (top the elevation −iωφ/g) the band term of a free
    /// waterline, under a plate moving with the wall (top = X_3) zero. Never with the closed forms
    /// (a body with plates has no pressure surfaces in the drift).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn waterline_under_plate(
        &mut self,
        phi: Complex64,
        top: Complex64,
        xb: &Vec3,
        n: &Vec3,
        dl: f64,
        shapes: Option<&[Vec3]>,
    ) {
        debug_assert!(!self.cf_);
        let D = self.vertical(xb, shapes);
        // P1_w / (rho g) and zeta_r
        let a = Complex64::new(0.0, -self.w_ / self.g_) * phi - D;
        let zr = top - D;
        let sq = (1.0 - n[2] * n[2]).max(1e-12).sqrt();
        let sw = -self.rho_ * self.g_ * (mean(a, zr) - 0.5 * mean(zr, zr)) * dl / sq;
        let xn = cross(*xb, *n);
        for d in 0..3 {
            self.F_[d] += sw * n[d];
            self.F_[3 + d] += sw * xn[d];
        }
        if let Some(s) = shapes.filter(|_| !self.xg_.is_empty()) {
            for (k, f) in self.Fg_.iter_mut().enumerate() {
                *f += sw * (s[k][0] * n[0] + s[k][1] * n[1] + s[k][2] * n[2]);
            }
        }
    }

    /// A point of the wetted (wetted = true) or pressure surface for the closed forms: x about
    /// O and n in body axes, its global z and weight.
    pub fn hydrostatic_point(&mut self, x: &Vec3, n: &Vec3, z: f64, weight: f64, wetted: bool) {
        self.cs_.add(x, n, z, weight, wetted);
    }
    /// The force and moment about the rotation centre R, then the moment about the moving R:
    /// o[9]; zo is the global z of O, c_rho_g the hydrostatic restoring of the wetted surface (its
    /// rho g terms, in N/m and N m; Hydrostatics::c_rho_g).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn finish(&self, zo: f64, c_rho_g: &[f64; 36], o: &mut [f64; 9]) {
        let (F, F1) = self.totals(zo, c_rho_g);
        let xr = [self.X_[0], self.X_[1], self.X_[2]];
        about_rotation_center(&F, &F1, &self.rc_, &xr, &self.xo_, o);
    }

    /// The force and moment about O alone (the hydrostatic part of the control-surface drift).
    pub fn finish_about_o(&self, zo: f64, c_rho_g: &[f64; 36]) -> [f64; 6] {
        self.totals(zo, c_rho_g).0
    }

    // the force and moment about O and the first-order force of the moving-point moment
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn totals(&self, zo: f64, c_rho_g: &[f64; 36]) -> ([f64; 6], Vec3c) {
        let mut F = self.F_;
        let mut F1 = self.F1_;
        if self.cf_ {
            // <a b> of the motions of O: 0..2 translation, 3..5 rotation
            let M = |i: usize, k: usize| {
                mean(
                    if i < 3 { self.xo_[i] } else { self.al_[i - 3] },
                    if k < 3 { self.xo_[k] } else { self.al_[k - 3] },
                )
            };
            let rg = self.rho_ * self.g_;
            let [mut f2, mut f3, mut f4] = self.cs_.terms(zo, M);
            if self.rc_ != Vec3::ZERO {
                // H about R: the second-order translation −H R of O
                let [t2, t3, t4] = self.cs_.translation([-self.HR_[0], -self.HR_[1], -self.HR_[2]]);
                f2 += t2;
                f3 += t3;
                f4 += t4;
            }
            F[2] -= rg * f2;
            F[3] += rg * f3;
            F[4] += rg * f4;
            // the first-order force of the moving-point moment: the dynamic force on the wetted
            // surface (without hydrostatics) and the rigid-mode restoring of the wetted surface
            for d in 0..3 {
                for k in 0..6 {
                    F1[d] -= c_rho_g[d * 6 + k] * self.X_[k];
                }
            }
        }
        (F, F1)
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn Hv(&self, v: &Vec3) -> Vec3 {
        let H_ = &self.H_;
        v3(H_[0][0] * v[0], H_[1][0] * v[0] + H_[1][1] * v[1], H_[2][0] * v[0] + H_[2][1] * v[1] + H_[2][2] * v[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{GeneralizedMode, ModePiece, ModeTerm};

    #[test]
    fn the_gradient_of_a_mode_is_its_derivative() {
        // a mode of several terms in all three components against central differences (step 1e-5:
        // the truncation 1e-10, the rounding 1e-11 of these values)
        let t = |component, px, py, pz, c| ModeTerm { component, px, py, pz, c };
        let gm = GeneralizedMode {
            name: "m".into(),
            parity: 1,
            pieces: vec![ModePiece {
                panels: vec![],
                terms: vec![
                    t(0, 1, 0, 1, 0.7),
                    t(1, 0, 2, 0, -1.3),
                    t(2, 1, 1, 1, 2.1),
                    t(2, 3, 0, 0, 0.4),
                    t(0, 0, 0, 0, 1.0),
                ],
            }],
        };
        let x = v3(0.3, -0.8, -0.45);
        let b = gm.gradient(&x, 1);
        let h = 1e-5;
        for k in 0..3 {
            let mut xp = x;
            let mut xm = x;
            xp[k] += h;
            xm[k] -= h;
            let (up, um) = (gm.displacement(&xp, 1), gm.displacement(&xm, 1));
            for i in 0..3 {
                let fd = (up[i] - um[i]) / (2.0 * h);
                assert!((fd - b[i][k]).abs() <= 1e-8, "{i} {k}: {fd} {}", b[i][k]);
            }
        }
    }

    #[test]
    fn the_band_under_a_plate_is_the_free_band_at_the_free_surface_and_none_on_a_cover() {
        // the band term with the plate's pressure at its top: with the top at the elevation
        // -i omega phi / g it is the free waterline's -rho g/2 <zeta_r^2> n / sqrt(1 - nz^2); with
        // the top at the wall's own X_3 (a plate moving with the wall) it vanishes
        let c = |re: f64, im: f64| Complex64::new(re, im);
        let x = [c(0.1, -0.2), c(0.05, 0.3), c(-0.4, 0.1), c(0.02, 0.01), c(-0.03, 0.04), c(0.01, -0.02)];
        let (rho, g, w) = (1025.0, 9.81, 0.9);
        let (phi, xb, n, dl) = (c(0.7, -1.3), v3(1.2, -0.4, 0.0), v3(0.6, -0.48, 0.64), 0.3);
        let shapes = [v3(0.1, 0.2, 0.5)];
        let new = || {
            let mut p = PressureDrift::new(&x, &v3(0.2, 0.0, -0.1), rho, g, w, false);
            p.generalized(&[c(0.2, -0.1)]);
            p
        };
        let (mut free, mut under, mut cover) = (new(), new(), new());
        free.waterline_with(phi, &xb, &n, dl, Some(&shapes));
        under.waterline_under_plate(phi, c(0.0, -w / g) * phi, &xb, &n, dl, Some(&shapes));
        let top = cover.vertical(&xb, Some(&shapes));
        cover.waterline_under_plate(phi, top, &xb, &n, dl, Some(&shapes));
        let scale = free.F_.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(scale > 0.0);
        for k in 0..6 {
            assert!((free.F_[k] - under.F_[k]).abs() <= 1e-12 * scale, "{k}: {} {}", free.F_[k], under.F_[k]);
            assert!(cover.F_[k] == 0.0, "{k}: {}", cover.F_[k]);
        }
        assert!((free.Fg_[0] - under.Fg_[0]).abs() <= 1e-12 * scale && cover.Fg_[0] == 0.0);
    }

    #[test]
    fn nanson_is_the_first_order_area_change() {
        // (E t1) × t2 + t1 × (E t2) = (tr E) n − Eᵀ n for t1 × t2 = n, and tangents of any normal
        let n = {
            let v = v3(0.3, -0.5, 0.8);
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v3(v[0] / l, v[1] / l, v[2] / l)
        };
        let (t1, t2) = tangents(&n);
        let c = cross(t1, t2);
        for k in 0..3 {
            assert!((c[k] - n[k]).abs() <= 1e-15);
        }
        let e: [[Complex64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|k| Complex64::new(0.1 * (i * 3 + k) as f64 - 0.3, 0.05 * k as f64))
        });
        let (e1, e2) = (mat_vec(&e, &t1), mat_vec(&e, &t2));
        let a = cross_c(&e1, &complex3(&t2));
        let b = cross_c(&complex3(&t1), &e2);
        let nn = nanson(&e, &n);
        for k in 0..3 {
            assert!((a[k] + b[k] - nn[k]).norm() <= 1e-15);
        }
    }
}
