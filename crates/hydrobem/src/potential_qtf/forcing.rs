//! The free-surface forcing of a pair of first-order waves (theory §12.4), its part of second
//! derivatives moved by the divergence theorem (§12.8), the incident plane waves and the one wall
//! (§12.14).

use num_complex::Complex64;

use super::I;
use crate::constants::PI;
use crate::model_codes::Domain;
use crate::problem::Problem;

/// The free-surface forcing of the pair (the pair coefficients of theory §12.4) of the fields x
/// of wave a and y of wave b: potentials, gradients (x, y, z) and γ = −ω²∂φ/∂z + g∂²φ/∂z² on z = 0.
#[inline]
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub(super) fn forcing(
    wa: f64,
    wb: f64,
    g: f64,
    sum: bool,
    xa: Complex64,
    ga: &[Complex64; 3],
    gma: Complex64,
    xb: Complex64,
    gb: &[Complex64; 3],
    gmb: Complex64,
) -> Complex64 {
    if sum {
        let o = wa + wb;
        -0.5 * I * o * (ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2])
            + (I * wa * xa * gmb + I * wb * xb * gma) / (4.0 * g)
    } else {
        let o = wa - wb;
        -0.5 * I * o * (ga[0] * gb[0].conj() + ga[1] * gb[1].conj() + ga[2] * gb[2].conj())
            + (I * wa * xa * gmb.conj() - I * wb * xb.conj() * gma) / (4.0 * g)
    }
}

/// The axis normal to the wall of a domain with one wall (0: the wall x = 0, fluid x > 0; 1: the
/// wall y = 0, fluid y > 0); None in the open sea. With a wall the problem is the open-sea problem
/// of the bodies and their mirror images under the waves and their mirror images, every field even
/// about the wall (theory §12.14): the free surface is integrated over the whole plane of that
/// problem and halved, its points beyond the wall evaluated at their mirror images.
pub(super) fn wall_axis(m: &Problem) -> Option<usize> {
    match m.opt.domain {
        Domain::XWall => Some(0),
        Domain::YWall => Some(1),
        _ => None,
    }
}

/// The heading of the mirror image of the wave of heading beta (radians) in the wall normal to
/// axis: π − β for the wall x = 0, −β for the wall y = 0.
pub(super) fn mirror_heading(axis: usize, beta: f64) -> f64 {
    if axis == 0 { PI - beta } else { -beta }
}

/// A first-order wave: its frequency, wavenumber, K = ω²/g and heading (radians); with a wall also
/// the heading of its mirror image (the incident wave of the domain the sum of both, as the
/// engine's first-order incident wave: a wave along the wall is its own image, doubled).
#[derive(Clone, Copy)]
pub(super) struct Wave {
    pub(super) w: f64,
    pub(super) k: f64,
    pub(super) kk: f64,
    pub(super) beta: f64,
    pub(super) image: Option<f64>,
}

/// A potential, its gradient (x, y, z) and γ at a point of z = 0.
pub(super) type Values = (Complex64, [Complex64; 3], Complex64);

impl Wave {
    /// The headings of the plane waves of the incident wave: the wave and its image.
    pub(super) fn headings(&self) -> Vec<f64> {
        match self.image {
            Some(b) => vec![self.beta, b],
            None => vec![self.beta],
        }
    }
    /// The plane wave of heading beta at (x, y, 0): its potential, gradient and γ.
    pub(super) fn plane(&self, beta: f64, g: f64, x: f64, y: f64) -> Values {
        let (c, s) = (beta.cos(), beta.sin());
        let phi = I * g / self.w * Complex64::from_polar(1.0, -self.k * (x * c + y * s));
        (
            phi,
            [-I * self.k * c * phi, -I * self.k * s * phi, self.kk * phi],
            g * (self.k * self.k - self.kk * self.kk) * phi,
        )
    }
    /// The incident potential at (x, y, 0) and its gradient and γ there (with a wall the sum of
    /// the wave and its image).
    pub(super) fn incident(&self, g: f64, x: f64, y: f64) -> Values {
        let (mut p, mut gr, mut gm) = self.plane(self.beta, g, x, y);
        if let Some(b) = self.image {
            let (p2, g2, m2) = self.plane(b, g, x, y);
            p += p2;
            gm += m2;
            for k in 0..3 {
                gr[k] += g2[k];
            }
        }
        (p, gr, gm)
    }
    /// The multipliers of the plane wave of heading beta without its phase factor (on the circle
    /// beyond R).
    pub(super) fn multipliers(&self, beta: f64, g: f64, phase: Complex64) -> Values {
        let a0 = I * g / self.w * phase;
        let (c, s) = (beta.cos(), beta.sin());
        (a0, [-I * self.k * c * a0, -I * self.k * s * a0, self.kk * a0], g * (self.k * self.k - self.kk * self.kk) * a0)
    }
}

/// The part −g ∇_h²φ_S of γ in the forcing (theory §12.8), moved by the divergence theorem so
/// that no second derivative is needed: with A, B the total first-order potentials of the pair (B
/// its conjugate for a difference), A', B' their disturbances and s = ±1 (sum, difference),
///   L = −(i/4)[ω_a A ∇_h²B' + s ω_b B ∇_h²A'],
///   ∫_D ψ L = (i/4) ∫_D [ω_a ∇_h(ψA)·∇_h B' + s ω_b ∇_h(ψB)·∇_h A']
///             − (i/4) ∮_∂D ψ [ω_a A ∂_ν B' + s ω_b B ∂_ν A'] dl, ν out of D.
pub(super) struct Lap {
    wa: f64,
    swb: f64,
    a: Complex64,
    ga: [Complex64; 2],
    gsa: [Complex64; 2],
    b: Complex64,
    gb: [Complex64; 2],
    gsb: [Complex64; 2],
}

/// The incident potential and gradient of a wave at a point, and its disturbance's.
pub(super) type Pair2 = (Complex64, [Complex64; 3]);

impl Lap {
    /// The transform of the pair at a point from the incident values (ia, ib) and the disturbances
    /// (sa, sb) of the waves a and b there: A = φ_I + φ_S of each (the horizontal gradients).
    pub(super) fn new(sum: bool, wa: f64, wb: f64, ia: Pair2, sa: Pair2, ib: Pair2, sb: Pair2) -> Lap {
        let (a, ga, gsa) = (ia.0 + sa.0, [ia.1[0] + sa.1[0], ia.1[1] + sa.1[1]], [sa.1[0], sa.1[1]]);
        let (b, gb, gsb) = (ib.0 + sb.0, [ib.1[0] + sb.1[0], ib.1[1] + sb.1[1]], [sb.1[0], sb.1[1]]);
        if sum {
            Lap { wa, swb: wb, a, ga, gsa, b, gb, gsb }
        } else {
            let c = |v: [Complex64; 2]| [v[0].conj(), v[1].conj()];
            Lap { wa, swb: -wb, a, ga, gsa, b: b.conj(), gb: c(gb), gsb: c(gsb) }
        }
    }
    /// The area term at a point of the free surface with ψ and ∇_h ψ.
    pub(super) fn area(&self, psi: Complex64, gpsi: [Complex64; 2]) -> Complex64 {
        let d = |u: [Complex64; 2], v: [Complex64; 2]| u[0] * v[0] + u[1] * v[1];
        let gpa = [self.a * gpsi[0] + psi * self.ga[0], self.a * gpsi[1] + psi * self.ga[1]];
        let gpb = [self.b * gpsi[0] + psi * self.gb[0], self.b * gpsi[1] + psi * self.gb[1]];
        0.25 * I * (self.wa * d(gpa, self.gsb) + self.swb * d(gpb, self.gsa))
    }
    /// The line term at a point of the boundary with ψ and the normal ν out of D.
    pub(super) fn line(&self, psi: Complex64, nu: [f64; 2]) -> Complex64 {
        let dn = |v: [Complex64; 2]| v[0] * nu[0] + v[1] * nu[1];
        -0.25 * I * psi * (self.wa * self.a * dn(self.gsb) + self.swb * self.b * dn(self.gsa))
    }
}
