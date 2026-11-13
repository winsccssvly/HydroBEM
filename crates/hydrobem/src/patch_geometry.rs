//! The geometry of one stored patch on [-1, 1]^2 (flat, B-spline, analytic or a trimmed element;
//! reference/inputs-meshes §2): used by the higher order and by the low-order control surfaces
//! of patch meshes (lo_geometry.rs). patch_geo is defined in ho_geometry.rs.
//!
//! A patch other than a flat one borrows its mesh (`mesh: Option<&'a Mesh>`), so PatchGeo (and the
//! HoGeom holding it) carry the lifetime of the mesh.

use crate::geometry::Vec3;
use crate::mesh::{Mesh, patch_eval};

/// The bilinear map of the four corners vt (at (-1, -1), (1, -1), (1, 1), (-1, 1)) at (u, v) in
/// [-1, 1]^2: the point x and the derivatives xu = dx/du, xv = dx/dv. Every flat patch and
/// bilinear analytic patch is evaluated here.
pub fn bilinear_map(vt: &[Vec3; 4], u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
    let n0 = 0.25 * (1.0 - u) * (1.0 - v);
    let n1 = 0.25 * (1.0 + u) * (1.0 - v);
    let n2 = 0.25 * (1.0 + u) * (1.0 + v);
    let n3 = 0.25 * (1.0 - u) * (1.0 + v);
    *x = n0 * vt[0] + n1 * vt[1] + n2 * vt[2] + n3 * vt[3];
    *xu = (0.25 * (1.0 - v)) * (vt[1] - vt[0]) + (0.25 * (1.0 + v)) * (vt[2] - vt[3]);
    *xv = (0.25 * (1.0 - u)) * (vt[3] - vt[0]) + (0.25 * (1.0 + u)) * (vt[2] - vt[1]);
}

/// The relative step into a patch from a point where its map is degenerate (a zero Jacobian: the
/// axis of a disc, a collapsed side, a waterline with a vanishing mapping slope): the normal there
/// is the limit normal at the point inside_point gives. Small enough that the normal of a smooth
/// map moves by a relative 1e-7 of its curvature over the patch, large enough that the Jacobian
/// there is far above the rounding of the map (about 1e-7 of its scale against 1e-16).
pub const DEGENERATE_STEP: f64 = 1e-7;

/// The point just inside the patch from (u, v) where the limit normal of a degenerate point is
/// taken: (u, v) (1 - DEGENERATE_STEP), towards the corner (1, 1) from the parametric centre.
pub fn inside_point(u: f64, v: f64) -> (f64, f64) {
    if u == 0.0 && v == 0.0 {
        return (DEGENERATE_STEP, DEGENERATE_STEP);
    }
    (u * (1.0 - DEGENERATE_STEP), v * (1.0 - DEGENERATE_STEP))
}

/// Geometry of one stored patch, parameters on [-1, 1]^2: a flat patch (or the flat quadrilateral
/// of a trimmed element) by the bilinear map of its corners vt, every other patch by patch_eval of
/// its mesh (B-spline, rational, analytic, trimmed: the one evaluator of the meshes).
#[derive(Clone, Debug)]
pub struct PatchGeo<'a> {
    pub flat: bool,
    pub analytic: bool,         // an analytic patch
    pub mesh: Option<&'a Mesh>, // not flat: the mesh of the element rec (evaluated by patch_eval)
    pub rec: i32,
    pub vt: [Vec3; 4], // flat patch vertices
}

impl Default for PatchGeo<'_> {
    fn default() -> Self {
        PatchGeo { flat: true, analytic: false, mesh: None, rec: 0, vt: [Vec3::ZERO; 4] }
    }
}

impl PatchGeo<'_> {
    pub fn eval(&self, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
        match self.mesh {
            Some(m) if !self.flat => patch_eval(m, self.rec, u, v, x, xu, xv),
            _ => bilinear_map(&self.vt, u, v, x, xu, xv),
        }
    }

    /// normal into the fluid: GDF vertex order for flat patches, the
    /// right-hand rule u -> v pointing into the body for spline patches
    pub fn nsign(&self) -> f64 {
        if self.flat { 1.0 } else { -1.0 }
    }
}
