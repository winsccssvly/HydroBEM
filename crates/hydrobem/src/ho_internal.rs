//! Internal types of the higher-order method (theory §7): the patch
//! geometry and discretization, the instances of the symmetry group, the
//! quadrature points and the functions shared by ho_geometry.rs,
//! ho_static.rs, ho_sheets.rs, ho_frequency.rs and ho_solver.rs, among them the spline values of a
//! solution at a point and the L2 projection of point values onto the basis.
//!
//! HoGeom borrows the problem it is built from (lifetime 'a): the geometry of the patches that are
//! not flat (PatchGeo::mesh) and the generalized modes (gmode) are references into it.

use crate::bspline::basis;
use crate::error::Result;
use crate::geometry::{Reflection, Vec3, cross, dot, norm, v3};
use crate::linalg::RealLu;
use crate::low_order::{Column, CsPoint, DamperSurf, PressSurf, SurfaceKind};
use crate::model_codes::PressureShape;
use crate::patch_geometry::{PatchGeo, inside_point};
use crate::problem::GeneralizedMode;
use num_complex::Complex64;

// ------------------------------------------------------------------ problem layout

#[derive(Clone, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct PatchDisc {
    pub nu: i32,
    pub nv: i32,
    pub ku: i32,
    pub kv: i32,
    pub iquo: i32,
    pub iqui: i32,
    pub Tu: Vec<f64>, // potential knots on [-1, 1]
    pub Tv: Vec<f64>,
}

impl PatchDisc {
    pub fn mu(&self) -> i32 {
        self.nu + self.ku - 1
    }
    pub fn mv(&self) -> i32 {
        self.nv + self.kv - 1
    }
    pub fn nb(&self) -> i32 {
        self.mu() * self.mv()
    }
}

/// A patch instance: a stored patch reflected by a local (mesh) reflection,
/// placed in the frame and possibly reflected in a wall. Its unknowns are those
/// of the row instance `rep` times chi[c][g] (g: element of the group).
#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub patch: i32,
    pub g: i32,
    pub r: Reflection, // local reflection
    pub body: i32,
    pub rep: i32,
    pub image: bool, // a wall image: moves as the mirror image of its body
    pub c: f64,      // rotation (yaw) into the frame
    pub s: f64,
    pub t: Vec3,       // translation into the frame
    pub w: Reflection, // wall reflection, applied last
    pub wt: f64,       // y offset after the wall reflection: a channel image y -> +-W - y
    pub f: f64,        // factor of the columns: the reflection factor of a channel image, else 1
}

impl Default for Inst {
    fn default() -> Self {
        Inst {
            patch: 0,
            g: 0,
            r: Reflection { sx: 1.0, sy: 1.0 },
            body: 0,
            rep: 0,
            image: false,
            c: 1.0,
            s: 0.0,
            t: Vec3::ZERO,
            w: Reflection { sx: 1.0, sy: 1.0 },
            wt: 0.0,
            f: 1.0,
        }
    }
}

/// generalized modes: 6 per body, then the pressure modes (usize: an array length)
pub const MAX_MODES: usize = 32;

/// The basis functions non-zero on one panel, ku * kv, at most 6 * 6: the potential orders are
/// in 2..6 (checked by mesh.rs)
pub const MAX_PANEL_BASIS: usize = 36;

/// A quadrature point on an instance.
#[derive(Clone, Copy, Debug)]
pub struct Pt {
    pub x: Vec3,
    pub n: Vec3,
    pub w: f64, // parametric weight (outer) or area weight (inner)
    pub inst: i32,
    pub iu: i32,
    pub iv: i32,
    pub bu: [f64; 8],
    pub bv: [f64; 8],
    pub xl: Vec3,             // body-local position (after the local reflection)
    pub nm: [f64; MAX_MODES], // generalized normals: rigid modes per body, then the pressure-mode shapes
    pub uf: f64,              // parameters
    pub vf: f64,
    // inner points: the columns and weights w U_a V_b of the non-zero basis functions
    pub nk: i32,
    pub col: [i32; MAX_PANEL_BASIS],
    pub wb: [f64; MAX_PANEL_BASIS],
}

impl Default for Pt {
    fn default() -> Self {
        Pt {
            x: Vec3::ZERO,
            n: Vec3::ZERO,
            w: 0.0,
            inst: 0,
            iu: 0,
            iv: 0,
            bu: [0.0; 8],
            bv: [0.0; 8],
            xl: Vec3::ZERO,
            nm: [0.0; MAX_MODES],
            uf: 0.0,
            vf: 0.0,
            nk: 0,
            col: [0; MAX_PANEL_BASIS],
            wb: [0.0; MAX_PANEL_BASIS],
        }
    }
}

/// A Gauss point of the mean waterline (an edge of a wet patch in z = 0).
#[derive(Clone, Copy, Debug, Default)]
pub struct WlPt {
    pub inst: i32,
    pub iu: i32,
    pub iv: i32,
    pub u: f64, // parameters, length weight
    pub v: f64,
    pub dl: f64,
    pub x: Vec3, // frame and body-local point and normal
    pub n: Vec3,
    pub xb: Vec3,
    pub nb: Vec3,
    pub plate: i32, // the body of the free-surface plate over the water beside it, -1: free surface
}

/// An evaluation point of the sheet rows: x (frame) with normal nx on a planar dipole patch,
/// its parameters and generalized normals (an outer Gauss point of a row instance).
#[derive(Clone, Copy, Debug)]
pub struct SheetPt {
    pub inst: i32,
    pub u: f64,
    pub v: f64,
    pub iu: i32,
    pub iv: i32,
    pub x: Vec3,
    pub n: Vec3,
    pub nm: [f64; MAX_MODES],
}

impl Default for SheetPt {
    fn default() -> Self {
        SheetPt { inst: 0, u: 0.0, v: 0.0, iu: 0, iv: 0, x: Vec3::ZERO, n: Vec3::ZERO, nm: [0.0; MAX_MODES] }
    }
}

/// A body output point (theory §4.1): the midpoints of the (ku + 1) x (kv + 1)
/// parametric subdivisions of every potential panel of the wetted and pressure
/// patches, body by body, patch by patch, panels and points with u
/// fastest; rec numbers the points per body and reflection.
#[derive(Clone, Copy, Debug)]
pub struct BPt {
    pub inst: i32,
    pub iu: i32,
    pub iv: i32,
    pub body: i32,
    pub rec: i32,
    pub given: bool, // a specified point (Body.body_points): ids (body, point, 0, 0)
    pub u: f64,      // parameters, J du dv of the subdivision
    pub v: f64,
    pub area: f64,
    pub x: Vec3, // frame
    pub n: Vec3,
    pub nm: [f64; MAX_MODES],
}

impl Default for BPt {
    fn default() -> Self {
        BPt {
            inst: 0,
            iu: 0,
            iv: 0,
            body: 0,
            rec: 0,
            given: false,
            u: 0.0,
            v: 0.0,
            area: 0.0,
            x: Vec3::ZERO,
            n: Vec3::ZERO,
            nm: [0.0; MAX_MODES],
        }
    }
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct HoGeom<'a> {
    pub geo: Vec<PatchGeo<'a>>,
    pub disc: Vec<PatchDisc>,
    pub refl: Vec<Reflection>,     // the group (frame reflections)
    pub ind: Vec<Vec<Reflection>>, // per body: the local reflection induced by each element
    pub chi: Vec<Vec<f64>>,
    pub inst: Vec<Inst>,
    pub rows: Vec<i32>,    // the row (representative) instances
    pub proj: Vec<RealLu>, // per patch: the factored mass matrix of the L2 projection
    pub dof0: Vec<i32>,    // DOF offset of each instance (that of its row instance)
    pub col0: Vec<i32>,    // column offset of each instance
    pub blk0: Vec<i32>,    // first (instance, panel) block of each instance
    pub Np: i32,
    pub G: i32,
    pub Ndof: i32,
    pub Ncol: i32,
    pub NB: i32,
    pub pbody: Vec<i32>,        // body of each stored patch
    pub walls: Vec<Reflection>, // wall reflections including the identity
    pub W: f64,                 // channel width (walls y = +-W/2), 0: none
    pub single: bool,           // a single body in open sea in its own horizontal frame
    pub outer: Vec<Pt>,         // outer: row instances; inner: all
    pub inner: Vec<Pt>,
    pub wl: Vec<WlPt>,    // the mean waterline of the bodies (not the images)
    pub bpts: Vec<BPt>,   // body output points
    pub sheets: bool,     // dipole patches present
    pub inner0: Vec<i32>, // first inner point of each (instance, panel): [inst][panel]
    pub mid: Vec<Pt>,     // 6 x 6 points of every block (mid tier of the static integrals)
    pub sfar: Vec<Pt>,    // 3 x 3 points of every block (far tier of the static integrals)
    pub bc: Vec<Vec3>,    // centre of each (instance, panel) block
    pub bs: Vec<f64>,     // size of each block (largest centre-corner distance)
    pub px: f64,          // the frame in global coordinates
    pub py: f64,
    pub yaw: f64,
    pub rc: Vec<Vec3>,     // rotation centre of each body (body axes)
    pub fp: Vec<Vec3>,     // field points in the frame (then the velocity stencils)
    pub Pu: i32,           // the field points of the problem (the first Pu of fp)
    pub fstep: f64,        // step of the velocity stencils, 0: no velocity
    pub cs: Vec<CsPoint>,  // the control-surface points
    pub cs_reps: Vec<i32>, // the points whose fields are evaluated (one per orbit)
    pub cs_rep: Vec<i32>,  // per point: its representative (index into cs_reps), x = g x_rep
    pub cs_g: Vec<i32>,
    pub cs_G: i32,                // the group used for the orbits (1: every point its own)
    pub cs_col: Vec<Vec<i32>>,    // per group element: the column of the reflected instance
    pub cs_inst: Vec<Vec<usize>>, // per group element: the reflected instance of each instance
    pub cs0: i32,                 // the first representative in fp, then 6 stencil points each
    pub csstep: f64,              // the step of their stencils
    pub wavemaker: bool,          // every body a wavemaker in the walls
    pub pdmp: Vec<i32>,           // per stored patch: its damper (index into dmp), else -1
    pub dmp: Vec<DamperSurf>,     // the dipole dampers
    pub depth: f64,
    // patch kinds (wet, interior free surface, pressure surface) and the
    // pressure surfaces and modes of the body (theory §9.3)
    pub kind: Vec<SurfaceKind>, // per stored patch: wet, interior free surface, pressure surface, lid or sheet
    pub pss: Vec<i32>,          // pressure surface of a stored patch, else -1
    pub ps: Vec<PressSurf>,
    pub pm_ps: Vec<i32>, // per pressure mode: surface, shape
    pub pm_shape: Vec<PressureShape>,
    pub press_out: Vec<i32>, // output index of the inner points of pressure patches, else -1
    pub Q: i32,
    pub lidx: Vec<i32>,    // damping lid of a stored patch, else -1
    pub lid_out: Vec<i32>, // output index of the inner points of lid patches, else -1
    pub L: i32,
    pub NM: i32, // generalized normals: 6 per body, the generalized modes, the pressure modes
    pub P0: i32, // index of the first pressure mode (6 NB + generalized modes)
    pub gmode: Vec<&'a GeneralizedMode>, // the generalized modes, body-major
    pub gm_body: Vec<i32>, // ... their bodies
    pub prec: Vec<i32>, // one-based element of each stored patch in its body mesh
    // compact copy of the inner points for the per-frequency wave rows: per
    // point x, n and w * generalized normals, per (instance, panel) block the
    // shared basis columns and per point their weights w U_a V_b
    pub wx: Vec<f64>, // [3 NQ], [3 NQ], [NM NQ], per block nk weights per point
    pub wn: Vec<f64>,
    pub wm: Vec<f64>,
    pub wb: Vec<f64>,
    pub bnk: Vec<i32>,
    pub bcol0: Vec<i32>,
    pub bcol: Vec<i32>,
    pub bwb0: Vec<i32>,
    // the same for the far rule of the wave part: order iqui - 1 per block
    pub fx: Vec<f64>,
    pub fn_: Vec<f64>, // the normals of the far points (`fn` is a keyword)
    pub fm: Vec<f64>,
    pub fwb: Vec<f64>,
    pub f0: Vec<i32>, // first far point of each block (size blocks + 1)
}

impl Default for HoGeom<'_> {
    fn default() -> Self {
        HoGeom {
            geo: Vec::new(),
            disc: Vec::new(),
            refl: Vec::new(),
            ind: Vec::new(),
            chi: Vec::new(),
            inst: Vec::new(),
            rows: Vec::new(),
            proj: Vec::new(),
            dof0: Vec::new(),
            col0: Vec::new(),
            blk0: Vec::new(),
            Np: 0,
            G: 1,
            Ndof: 0,
            Ncol: 0,
            NB: 1,
            pbody: Vec::new(),
            walls: Vec::new(),
            W: 0.0,
            single: false,
            outer: Vec::new(),
            inner: Vec::new(),
            wl: Vec::new(),
            bpts: Vec::new(),
            sheets: false,
            inner0: Vec::new(),
            mid: Vec::new(),
            sfar: Vec::new(),
            bc: Vec::new(),
            bs: Vec::new(),
            px: 0.0,
            py: 0.0,
            yaw: 0.0,
            rc: Vec::new(),
            fp: Vec::new(),
            Pu: 0,
            fstep: 0.0,
            cs: Vec::new(),
            cs_reps: Vec::new(),
            cs_rep: Vec::new(),
            cs_g: Vec::new(),
            cs_G: 1,
            cs_col: Vec::new(),
            cs_inst: Vec::new(),
            cs0: 0,
            csstep: 0.0,
            wavemaker: false,
            pdmp: Vec::new(),
            dmp: Vec::new(),
            depth: f64::INFINITY,
            kind: Vec::new(),
            pss: Vec::new(),
            ps: Vec::new(),
            pm_ps: Vec::new(),
            pm_shape: Vec::new(),
            press_out: Vec::new(),
            Q: 0,
            lidx: Vec::new(),
            lid_out: Vec::new(),
            L: 0,
            NM: 6,
            P0: 6,
            gmode: Vec::new(),
            gm_body: Vec::new(),
            prec: Vec::new(),
            wx: Vec::new(),
            wn: Vec::new(),
            wm: Vec::new(),
            wb: Vec::new(),
            bnk: Vec::new(),
            bcol0: Vec::new(),
            bcol: Vec::new(),
            bwb0: Vec::new(),
            fx: Vec::new(),
            fn_: Vec::new(),
            fm: Vec::new(),
            fwb: Vec::new(),
            f0: Vec::new(),
        }
    }
}

/// The points of a block (instance, panel) of the wave rows by one rule: the point range q0..q1,
/// their coordinates x and normals n ([3 per point]), their weighted generalized normals m ([NM per
/// point]) and the basis weights wb of the block's nk columns per point, from wb0.
pub struct BlockRule<'h> {
    pub q0: usize,
    pub q1: usize,
    pub wb: &'h [f64],
    pub wb0: usize,
    pub x: &'h [f64],
    pub n: &'h [f64],
    pub m: &'h [f64],
}

impl HoGeom<'_> {
    /// The points of block bk by the far rule (far) or the inner rule.
    pub fn block_rule(&self, bk: usize, far: bool) -> BlockRule<'_> {
        if far {
            let q0 = self.f0[bk] as usize;
            let (q1, wb0) = (self.f0[bk + 1] as usize, q0 * self.bnk[bk] as usize);
            BlockRule { q0, q1, wb: &self.fwb, wb0, x: &self.fx, n: &self.fn_, m: &self.fm }
        } else {
            let (q0, q1) = (self.inner0[bk] as usize, self.inner0[bk + 1] as usize);
            BlockRule { q0, q1, wb: &self.wb, wb0: self.bwb0[bk] as usize, x: &self.wx, n: &self.wn, m: &self.wm }
        }
    }

    /// Whether block bk lies beyond WAVE_FAR block sizes from the nearer of the far channel images
    /// (y -+ 2W) of the point x: the far rule for the channel terms (theory §8.5).
    pub fn far_from_wall_images(&self, bk: usize, x: &Vec3) -> bool {
        let (c, w) = (&self.bc[bk], self.W);
        let rx = x[0] - c[0];
        let ry = (x[1] - 2.0 * w - c[1]).abs().min((x[1] + 2.0 * w - c[1]).abs());
        (rx * rx + ry * ry) / (self.bs[bk] * self.bs[bk]) > WAVE_FAR * WAVE_FAR
    }

    /// The near channel images in a row: the columns (a, and b when given) and the generalized
    /// normals (m, NM per instance) of every wall-image instance scaled by its reflection factor.
    /// Returns the factor Rc of the farther images (that of the first instance moved by a wall
    /// translation; 1 without one), which scales the terms of channel_far.
    pub fn scale_wall_images(&self, a: &mut [Complex64], mut b: Option<&mut [Complex64]>, m: &mut [Complex64]) -> f64 {
        let nm = self.NM as usize;
        for ii in 0..self.inst.len() {
            let f = self.inst[ii].f;
            if f == 1.0 {
                continue;
            }
            let c0 = self.col0[ii] as usize;
            for c in c0..c0 + self.disc_of(ii as i32).nb() as usize {
                a[c] *= f;
                if let Some(b) = b.as_deref_mut() {
                    b[c] *= f;
                }
            }
            for x in &mut m[ii * nm..(ii + 1) * nm] {
                *x *= f;
            }
        }
        self.inst.iter().find(|i| i.wt != 0.0).map_or(1.0, |i| i.f)
    }

    pub fn any_free_damper(&self) -> bool {
        for d in &self.dmp {
            if d.free {
                return true;
            }
        }
        false
    }
    pub fn panels_of(&self, p: i32) -> i32 {
        let d = &self.disc[p as usize];
        d.nu * d.nv
    }
    pub fn disc_of(&self, ii: i32) -> &PatchDisc {
        &self.disc[self.inst[ii as usize].patch as usize]
    }
    /// sign of a generalized normal: +1 for the rigid and generalized modes (normal velocity
    /// n_j), -1 for a pressure mode (u = -n_k on its surface, forces with the opposite sign)
    pub fn usign(&self, m: i32) -> f64 {
        if m < self.P0 { 1.0 } else { -1.0 }
    }
    pub fn kind_of(&self, ii: i32) -> SurfaceKind {
        self.kind[self.inst[ii as usize].patch as usize]
    }
    pub fn lid_of(&self, ii: i32) -> i32 {
        self.lidx[self.inst[ii as usize].patch as usize]
    }
    pub fn z0(&self, ii: i32) -> bool {
        self.kind_of(ii) == SurfaceKind::Press && self.ps[self.pss[self.inst[ii as usize].patch as usize] as usize].z0
    }
}

/// Point, frame normal (into the fluid) and area element of instance i at (u, v);
/// xb, nb: the body-local point and normal (after the local reflection).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn inst_eval(
    h: &HoGeom,
    in_: &Inst,
    u: f64,
    v: f64,
    x: &mut Vec3,
    n: &mut Vec3,
    J: &mut f64,
    xb: Option<&mut Vec3>,
    nb: Option<&mut Vec3>,
) {
    let geo = &h.geo[in_.patch as usize];
    let (mut xl, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    geo.eval(u, v, &mut xl, &mut xu, &mut xv);
    let mut nn = geo.nsign() * cross(xu, xv);
    *J = norm(nn);
    if !(*J > 0.0) {
        // a degenerate edge (zero Jacobian, e.g. the axis of a disc or a waterline with a
        // vanishing mapping slope): the limit normal from just inside the patch
        let (mut x2, mut xu2, mut xv2) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let (ui, vi) = inside_point(u, v);
        geo.eval(ui, vi, &mut x2, &mut xu2, &mut xv2);
        nn = geo.nsign() * cross(xu2, xv2);
        nn = (1.0 / norm(nn)) * nn;
    } else {
        nn = (1.0 / *J) * nn;
    }
    let a = v3(in_.r.sx * xl[0], in_.r.sy * xl[1], xl[2]);
    let an = v3(in_.r.sx * nn[0], in_.r.sy * nn[1], nn[2]);
    *x = v3(
        in_.w.sx * (in_.c * a[0] - in_.s * a[1] + in_.t[0]),
        in_.w.sy * (in_.s * a[0] + in_.c * a[1] + in_.t[1]) + in_.wt,
        a[2] + in_.t[2],
    );
    *n = v3(in_.w.sx * (in_.c * an[0] - in_.s * an[1]), in_.w.sy * (in_.s * an[0] + in_.c * an[1]), an[2]);
    if let Some(xb) = xb {
        *xb = a;
    }
    if let Some(nb) = nb {
        *nb = an;
    }
}

/// index of the (instance, panel) block in inner0
#[inline]
pub fn block_of(h: &HoGeom, ii: i32, iu: i32, iv: i32) -> i32 {
    h.blk0[ii as usize] + iv * h.disc_of(ii).nu + iu
}

pub const MID_ORDER: i32 = 6;
/// the far tier of the static integrals beyond STATIC_FAR sizes
pub const FAR_ORDER: i32 = 3;

// ------------------------------------------------------------------ static integrals

/// Static integrals of one target point against every instance: per column
/// (instance, basis) the Rankine source and dipole of the image sum (and the
/// omega = inf variant) and the corrections of the log singularity relative to
/// the inner rule; per (instance, mode) the same for the source with u = n_j.
#[derive(Clone, Debug, Default)]
pub struct StaticRow {
    pub rs: Vec<f64>, // [Ncol]; ls, ld empty once compressed
    pub rd: Vec<f64>,
    pub ls: Vec<f64>,
    pub ld: Vec<f64>,
    pub rsi: Vec<f64>,
    pub rdi: Vec<f64>,
    pub ms: Vec<f64>, // [inst * NM + mode]; ml empty once compressed
    pub ml: Vec<f64>,
    pub msi: Vec<f64>,
    // the log corrections, nonzero for the near columns only, kept sparse:
    // the columns and their (ls, ld), the (instance, mode) entries and their ml
    pub lcol: Vec<i32>,
    pub mcol: Vec<i32>,
    pub lsv: Vec<f64>,
    pub ldv: Vec<f64>,
    pub mlv: Vec<f64>,
}
// StaticRow::compress is in ho_static.rs.

// ------------------------------------------------------------------ per frequency

/// Surface gradient from the parametric derivatives and the tangents (frame).
#[inline]
pub fn surface_gradient(xu: &Vec3, xv: &Vec3, fu: Complex64, fv: Complex64, g: &mut [Complex64; 3]) {
    let a = dot(*xu, *xu);
    let b = dot(*xu, *xv);
    let c = dot(*xv, *xv);
    let det = a * c - b * b;
    let p = (c * fu - b * fv) / det;
    let q = (a * fv - b * fu) / det;
    for k in 0..3 {
        g[k] = p * xu[k] + q * xv[k];
    }
}

/// Tangents of an instance at (u, v) in the frame: the local reflection, the yaw and the wall
/// reflection of the instance applied.
#[inline]
pub fn inst_tangents(h: &HoGeom, in_: &Inst, u: f64, v: f64, xu: &mut Vec3, xv: &mut Vec3) {
    let mut xl = Vec3::ZERO;
    h.geo[in_.patch as usize].eval(u, v, &mut xl, xu, xv);
    let tr = |a0: &Vec3| -> Vec3 {
        let a = v3(in_.r.sx * a0[0], in_.r.sy * a0[1], a0[2]);
        v3(in_.w.sx * (in_.c * a[0] - in_.s * a[1]), in_.w.sy * (in_.s * a[0] + in_.c * a[1]), a[2])
    };
    *xu = tr(xu);
    *xv = tr(xv);
}

/// distance (in block sizes) from the image of the target beyond which the
/// wave part uses the far rule (the Gauss rule of order q_i - 1: at 3 sizes rho = 7.6, about 5e-6
/// per panel; it took about 40 % off the frequency time, theory §7.2)
pub const WAVE_FAR: f64 = 3.0;

/// sigma_g(mode): sign of generalized normal m under the reflection (sx, sy)
/// (the pressure modes: the parity of their shape)
/// (gi: element of the group, acting on the body of the mode by its induced local reflection)
#[inline]
pub fn mode_sign(h: &HoGeom, gi: i32, m: i32) -> f64 {
    let gi = gi as usize;
    if m >= h.P0 {
        let k = (m - h.P0) as usize;
        let r = &h.ind[h.ps[h.pm_ps[k] as usize].body as usize][gi];
        let sh = h.pm_shape[k];
        return if sh == PressureShape::SignX {
            r.sx
        } else if sh == PressureShape::SignY {
            r.sy
        } else {
            1.0
        };
    }
    if m >= 6 * h.NB {
        // a generalized mode: its parity (the rigid mode whose reflection it shares)
        let k = (m - 6 * h.NB) as usize;
        let r = &h.ind[h.gm_body[k] as usize][gi];
        let s = [r.sx, r.sy, 1.0, r.sy, r.sx, r.sx * r.sy];
        return s[(h.gmode[k].parity - 1) as usize];
    }
    let r = &h.ind[(m / 6) as usize][gi];
    let s = [r.sx, r.sy, 1.0, r.sy, r.sx, r.sx * r.sy];
    s[(m % 6) as usize]
}

/// the generalized mode index (0..NM-1) of a radiation column or coordinate
#[inline]
pub fn mode_index(h: &HoGeom, c: &Column) -> i32 {
    if c.mode > 0 {
        return 6 * c.body + c.mode - 1;
    }
    if c.mode < 0 {
        return 6 * h.NB + c.gm;
    }
    let mut k = 0;
    for q in 0..h.ps.len() {
        if q as i32 == c.ps {
            return h.P0 + k + (c.coord - h.ps[q].coord0);
        }
        k += h.ps[q].shapes.len() as i32;
    }
    unreachable!("a radiation column of pressure surface {} that the problem does not have", c.ps);
}

// ------------------------------------------------------------------ thin sheets

/// Per sheet row: for every column the second normal derivative H = int B d2G/dn_x dn_xi
/// (wet and sheet columns), the source normal derivative Kc = int B dG/dn_x (wet columns),
/// and per (instance, mode) Mn = int n_mode dG/dn_x (wet instances).
#[derive(Clone, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct SheetRowData {
    pub H: Vec<Complex64>,
    pub Kc: Vec<Complex64>,
    pub Mn: Vec<Complex64>,
}

/// An evaluation point of the sheet rows with its test functions (rows, weights) and the
/// Rankine part of its integrals (ordinary and zero frequency; omega = inf).
#[derive(Clone, Debug, Default)]
pub struct SheetEval {
    pub sp: SheetPt,
    pub rw: Vec<(i32, f64)>,
    pub w: f64,      // the weight of the point (rw: w times the test functions)
    pub plate: bool, // a point of a free-surface plate: a sheet row at omega = inf only (no reg)
    pub reg: SheetRowData,
    pub inf: SheetRowData,
}

const Z: Complex64 = Complex64::new(0.0, 0.0);

/// The total first-order potential of one heading at a point of an instance:
/// the B-spline coefficients cv, its value and parametric derivatives (u, v).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn spline_values(
    h: &HoGeom,
    ii: i32,
    cv: &[Complex64],
    u: f64,
    v: f64,
    iu: i32,
    iv: i32,
    f: &mut Complex64,
    fu: &mut Complex64,
    fv: &mut Complex64,
) {
    let d = h.disc_of(ii);
    let mut Nu = [0.0f64; 8];
    let mut dNu = [0.0f64; 8];
    let mut Nv = [0.0f64; 8];
    let mut dNv = [0.0f64; 8];
    basis(&d.Tu, d.ku, iu + d.ku - 1, u, &mut Nu, Some(&mut dNu));
    basis(&d.Tv, d.kv, iv + d.kv - 1, v, &mut Nv, Some(&mut dNv));
    *f = Z;
    *fu = Z;
    *fv = Z;
    let c0 = h.col0[ii as usize];
    let mu = d.mu();
    for b in 0..d.kv as usize {
        for a in 0..d.ku as usize {
            let c = cv[(c0 + (iv + b as i32) * mu + iu + a as i32) as usize];
            *f += Nu[a] * Nv[b] * c;
            *fu += dNu[a] * Nv[b] * c;
            *fv += Nu[a] * dNv[b] * c;
        }
    }
}

/// L2 projection of point values on the inner points of an instance onto its
/// basis (parametric mass matrix with the area weights), with the factored mass matrix of its
/// patch (build_projections).
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn project(h: &HoGeom, ii: i32, vals: &[Complex64], coef: &mut Vec<Complex64>) -> Result<()> {
    let d = &h.disc[h.inst[ii as usize].patch as usize];
    let nb = d.nb() as usize;
    let mu = d.mu();
    let mut rhs = vec![Z; nb];
    let first = block_of(h, ii, 0, 0) as usize;
    let last = first + (d.nu * d.nv) as usize;
    for bk in first..last {
        for qi in h.inner0[bk] as usize..h.inner0[bk + 1] as usize {
            let q = &h.inner[qi];
            for b in 0..d.kv as usize {
                for a in 0..d.ku as usize {
                    rhs[((q.iv + b as i32) * mu + q.iu + a as i32) as usize] += vals[qi] * (q.w * q.bu[a] * q.bv[b]);
                }
            }
        }
    }
    h.proj[h.inst[ii as usize].patch as usize].solve(&mut rhs)?;
    *coef = rhs;
    Ok(())
}
