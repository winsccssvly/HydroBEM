//! The data types and kernels of the solve: the real type `Real` of the low-order kernels, the
//! low-order `Geometry` (panels, surface kinds, pressure and damper surfaces), the static
//! influence data `Static`, the per-frequency context `FreqCtx`, the `Plan` and the `Output`
//! (the last three shared with the higher-order method, ho_*.rs), the pair kernel of the
//! free-surface Green function, the channel kernels, the incident wave and the far-field
//! (Kochin) kernels, and the profiling counters. The steps that use them are in lo_geometry,
//! lo_static, lo_frequency, lo_velocity (low order, driven by solver.rs), motion, solve_common
//! (the plan set-up) and tanks.
//!
//! The real type of the low-order kernels (float in production, double in the diagnostic mode)
//! is a generic `R: Real`; why single precision suffices in the low order (well-conditioned
//! systems, the production path within 1e-5 of the double-precision one): theory §13.1 and §13.2. `Real` (below) is implemented for f32 and f64;
//! it is a `num_complex::ComplexFloat<Real = Self>`, so R has the arithmetic operators and the
//! float functions (sqrt, abs, ...) and `Complex<R>` its complex arithmetic. A generic body
//! converts with `R::from_f64(x)` (a double rounded to R), `r.as_f64()` (R widened),
//! `R::c_from(z)` (a Complex64 rounded to `Complex<R>`) and `R::c_as_cd(c)` (widened); the
//! LU is `R::lu_solve(n, a, nrhs, b, work)` (crate::linalg::lu_solve_c64 / _c128) and the
//! C = A B product `R::gemm` (crate::linalg::gemm). The names as_f64 and from_f64 avoid
//! NumCast::to_f64 / from of ComplexFloat's supertraits.

use crate::model_codes::{PressureShape, QtfHeadings};
use std::fmt::Debug;
use std::time::Instant;

use num_complex::{Complex, Complex32, Complex64, ComplexFloat};

use crate::constants::{GAMMA, LOG2, PI};
use crate::env_hooks::env_set;
use crate::error::Result;
use crate::geometry::{Reflection, Vec3, v3};
use crate::green_channel::ChannelTail;
use crate::green_fin::{
    FinPoint, FiniteDepth, FreqMode, Kern, fin4_applies, finite_kernel, finite_kernel_hess, finite_kernel4,
    finite_series, finite_series_hess, propagating_factor, propagating_products,
};
use crate::green_inf::{
    WaveKernel, direct4_applies, near4_applies, wave_kernel, wave_kernel_direct4, wave_kernel_near4, wave_x_over_x,
};
use crate::panel_integrals::{PanelGeom, pair_derivatives};
use crate::progress::Work;
use crate::quadratic_qtf::QtfSamples;

// Notation of the solve (both methods; see also docs/development/architecture.md):
//   F ordinary frequencies (fi its index; -1 a limit), H headings (h, hh), D coordinates (6 per
//   body, its generalized and pressure modes), K or nrad solved radiation columns, nrhs = K + H,
//   P field points, L lid panels (output), Q pressure-surface panels (output), NK Kochin angles,
//   Nb body output points, B bodies.
//   Low order: Nf all panels in the frame (with reflections and wall images), Ns the panels of
//   the stored sector, G the reflection group, NC the solved symmetry characters (chi).
//   g the Geometry (lo) or a group element (ho), pl the Plan, fc the FreqCtx, h the HoGeom (ho).
//   In a FreqCtx, K = omega^2 / g and k the wave number; elsewhere K counts the radiation columns.
//   S and D in the kernels: the source and dipole influences.

// ------------------------------------------------------------------ the real type of the generic kernels

/// The real type R of the generic kernels (f32 in production, f64 in the diagnostic mode).
pub trait Real: ComplexFloat<Real = Self> + Default + Debug + Send + Sync + PartialOrd + Into<f64> + 'static {
    /// x rounded to R.
    fn from_f64(v: f64) -> Self;
    /// self widened to a double.
    fn as_f64(self) -> f64;
    /// z rounded to `Complex<R>` (each part).
    fn c_from(z: Complex64) -> Complex<Self>;
    /// c widened to a Complex64.
    fn c_as_cd(c: Complex<Self>) -> Complex64;
    /// The LU solve of linalg.rs in this precision: A X = B in place
    /// (crate::linalg::lu_solve_c64 / lu_solve_c128).
    fn lu_solve(n: i32, a: &mut [Complex<Self>], nrhs: i32, b: &mut [Complex<Self>], work: Work<'_>) -> Result<()>;
    /// lu_factor of linalg.rs in this precision (the factors kept for more right-hand sides).
    fn lu_factor(n: i32, a: &mut [Complex<Self>], ipiv: &mut [usize], work: Work<'_>) -> Result<()>;
    /// lu_substitute of linalg.rs in this precision.
    fn lu_substitute(
        n: i32,
        a: &[Complex<Self>],
        ipiv: &[usize],
        nrhs: i32,
        b: &mut [Complex<Self>],
        work: Work<'_>,
    ) -> Result<()>;
    /// C = A B, column-major, no transposes (crate::linalg::gemm: alpha 1, beta 0); m x k times
    /// k x n into m x n, the matrices with the leading dimensions lda, ldb, ldc.
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[Complex<Self>],
        lda: usize,
        b: &[Complex<Self>],
        ldb: usize,
        c: &mut [Complex<Self>],
        ldc: usize,
    );
}

impl Real for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
    fn c_from(z: Complex64) -> Complex32 {
        Complex32::new(z.re as f32, z.im as f32)
    }
    fn c_as_cd(c: Complex32) -> Complex64 {
        Complex64::new(c.re as f64, c.im as f64)
    }
    fn lu_factor(n: i32, a: &mut [Complex32], ipiv: &mut [usize], work: Work<'_>) -> Result<()> {
        crate::linalg::lu_factor(n, a, ipiv, work)
    }
    fn lu_substitute(
        n: i32,
        a: &[Complex32],
        ipiv: &[usize],
        nrhs: i32,
        b: &mut [Complex32],
        work: Work<'_>,
    ) -> Result<()> {
        crate::linalg::lu_substitute(n, a, ipiv, nrhs, b, work)
    }
    fn lu_solve(n: i32, a: &mut [Complex32], nrhs: i32, b: &mut [Complex32], work: Work<'_>) -> Result<()> {
        crate::linalg::lu_solve_c64(n, a, nrhs, b, work)
    }
    fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[Complex32],
        lda: usize,
        b: &[Complex32],
        ldb: usize,
        c: &mut [Complex32],
        ldc: usize,
    ) {
        crate::linalg::gemm(m, n, k, a, lda, b, ldb, c, ldc)
    }
}

impl Real for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn as_f64(self) -> f64 {
        self
    }
    fn c_from(z: Complex64) -> Complex64 {
        z
    }
    fn c_as_cd(c: Complex64) -> Complex64 {
        c
    }
    fn lu_factor(n: i32, a: &mut [Complex64], ipiv: &mut [usize], work: Work<'_>) -> Result<()> {
        crate::linalg::lu_factor(n, a, ipiv, work)
    }
    fn lu_substitute(
        n: i32,
        a: &[Complex64],
        ipiv: &[usize],
        nrhs: i32,
        b: &mut [Complex64],
        work: Work<'_>,
    ) -> Result<()> {
        crate::linalg::lu_substitute(n, a, ipiv, nrhs, b, work)
    }
    fn lu_solve(n: i32, a: &mut [Complex64], nrhs: i32, b: &mut [Complex64], work: Work<'_>) -> Result<()> {
        crate::linalg::lu_solve_c128(n, a, nrhs, b, work)
    }
    fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        c: &mut [Complex64],
        ldc: usize,
    ) {
        crate::linalg::gemm(m, n, k, a, lda, b, ldb, c, ldc)
    }
}

/// Optional phase timing on stderr (environment variable HB_PROFILE).
#[derive(Clone, Debug)]
pub struct Profile {
    pub on: bool,
    pub t: Instant,
}

impl Default for Profile {
    fn default() -> Self {
        Profile { on: env_set("HB_PROFILE"), t: Instant::now() }
    }
}

impl Profile {
    pub fn new() -> Profile {
        Profile::default()
    }
    pub fn mark(&mut self, what: &str) {
        let now = Instant::now();
        if self.on {
            eprintln!("hb profile: {:<28} {:9.4} s", what, (now - self.t).as_secs_f64());
        }
        self.t = now;
    }
}

// ------------------------------------------------------------------ geometry

/// A point of a control surface: the frame point, the body-local point and the
/// normal away from the body (body axes), its weight (area or length), its part
/// (0: submerged S_c, 1: free surface S_f, 2: the line CL), its body.
#[derive(Clone, Copy, Debug, Default)]
pub struct CsPoint {
    pub x: Vec3,
    pub xb: Vec3,
    pub nb: Vec3,
    pub w: f64,
    pub part: i32,
    pub body: i32,
}

/// The kind of the surface of a panel or patch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SurfaceKind {
    /// the wetted surface of a body
    #[default]
    Wet,
    /// an interior free surface (irregular-frequency removal, theory §5.1)
    Interior,
    /// a damping lid (theory §9.1)
    Lid,
    /// a thin sheet or dipole damper (theory §9.5)
    Sheet,
    /// a prescribed-pressure surface (theory §9.3)
    Press,
    /// a free-surface plate of a body (theory §9.7): moves with the body, lies in z = 0
    Plate,
}

/// A prescribed-pressure surface (theory §9.3): its body, the global
/// coordinates of its modes and their shapes; z0: in the plane z = 0.
#[derive(Clone, Debug)]
pub struct PressSurf {
    pub body: i32,
    pub coord0: i32,
    pub id: i32,                    // id: one-based surface number within the body
    pub shapes: Vec<PressureShape>, // per mode
    pub z0: bool,
    pub chamber: bool,
    pub numeric: bool,
}

impl Default for PressSurf {
    fn default() -> Self {
        PressSurf { body: -1, coord0: 0, id: 0, shapes: Vec::new(), z0: false, chamber: false, numeric: false }
    }
}

/// Shape n(x, y) of a pressure mode at the expanded body-local centroid.
#[inline]
pub fn press_shape(shape: PressureShape, lc: &Vec3) -> f64 {
    if shape == PressureShape::SignX {
        return if lc[0] > 0.0 {
            1.0
        } else if lc[0] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
    if shape == PressureShape::SignY {
        return if lc[1] > 0.0 {
            1.0
        } else if lc[1] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
    1.0
}

/// A dipole damping surface: epsilon and its condition.
#[derive(Clone, Copy, Debug)]
pub struct DamperSurf {
    pub eps: f64,
    pub free: bool,
    pub porous: bool,
}

impl Default for DamperSurf {
    fn default() -> Self {
        DamperSurf { eps: 0.0, free: true, porous: false }
    }
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct Geometry {
    pub Ns: i32,
    pub G: i32,
    pub Nf: i32,
    pub refl: Vec<Reflection>,  // symmetry group of the frame (e, sx, sy, sxsy)
    pub chi: Vec<Vec<f64>>,     // chi[c][g], only the characters that are solved
    pub pg: Vec<PanelGeom>,     // Nf panels
    pub raw: Vec<[Vec3; 4]>,    // their vertices as given (not projected), reflected and placed
    pub sec: Vec<i32>,          // Ns sector panels (indices into pg)
    pub img: Vec<i32>,          // img[g * Ns + k]: panel g(sec[k])
    pub body: Vec<i32>,         // body of each panel (-1: external lid)
    pub nj: Vec<[f64; 6]>,      // generalized normals, body axes about R (0 off the wetted surface)
    pub kind: Vec<SurfaceKind>, // wet, interior free surface (theory §5.1), lid (theory §9.1) or sheet (theory §9.5)
    pub lid: Vec<i32>,          // lid index of lid panels, else -1
    pub lid_out: Vec<i32>,      // output index of real lid panels, else -1
    pub mirror: Vec<i8>,        // wall image of a panel
    pub rec: Vec<i32>,          // the element of the body mesh (zero-based), else -1
    pub lref: Vec<Reflection>,  // the local reflection of the element
    pub walls: Vec<Reflection>, // wall reflections including the identity
    pub yaw: f64,               // frame: x_global = Rz(yaw) x_frame + (px, py, 0), yaw in degrees
    pub px: f64,
    pub py: f64,
    pub fp: Vec<Vec3>,        // field points in the frame
    pub cs: Vec<CsPoint>,     // the control-surface points
    pub cs_reps: Vec<i32>,    // the control-surface points whose fields are computed
    pub cs_rep: Vec<i32>,     // per point: its representative (index into cs_reps) and the
    pub cs_g: Vec<i32>,       // group element with x = g x_rep
    pub L: i32,               // lid panels (output)
    pub sheets: bool,         // thin sheets present
    pub plates: bool,         // free-surface plates present
    pub source: bool,         // source formulation for the velocities
    pub wavemaker: bool,      // every body a wavemaker in the walls
    pub ps: Vec<PressSurf>,   // pressure surfaces
    pub psurf: Vec<i32>,      // pressure surface of each panel, else -1
    pub lc: Vec<Vec3>,        // expanded body-local centroid of pressure panels
    pub press_out: Vec<i32>,  // output index of real pressure panels, else -1
    pub Q: i32,               // pressure panels (output)
    pub gn: Vec<Vec<f64>>,    // generalized modes: n . (u, v, w) per panel, [mode][Nf]
    pub gw: Vec<Vec<f64>>,    // ... the vertical displacement w per panel
    pub dmp: Vec<DamperSurf>, // dipole dampers
    pub damper: Vec<i32>,     // damper of each sheet panel, else -1
    pub W: f64,               // channel width (walls y = +-W/2), 0: none
    pub Rc: f64,              // channel: the factor of the wall images
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            Ns: 0,
            G: 1,
            Nf: 0,
            refl: Vec::new(),
            chi: Vec::new(),
            pg: Vec::new(),
            raw: Vec::new(),
            sec: Vec::new(),
            img: Vec::new(),
            body: Vec::new(),
            nj: Vec::new(),
            kind: Vec::new(),
            lid: Vec::new(),
            lid_out: Vec::new(),
            mirror: Vec::new(),
            rec: Vec::new(),
            lref: Vec::new(),
            walls: Vec::new(),
            yaw: 0.0,
            px: 0.0,
            py: 0.0,
            fp: Vec::new(),
            cs: Vec::new(),
            cs_reps: Vec::new(),
            cs_rep: Vec::new(),
            cs_g: Vec::new(),
            L: 0,
            sheets: false,
            plates: false,
            source: false,
            wavemaker: false,
            ps: Vec::new(),
            psurf: Vec::new(),
            lc: Vec::new(),
            press_out: Vec::new(),
            Q: 0,
            gn: Vec::new(),
            gw: Vec::new(),
            dmp: Vec::new(),
            damper: Vec::new(),
            W: 0.0,
            Rc: 1.0,
        }
    }
}

impl Geometry {
    /// The panel g j of every panel j under every group element g, [g * Nf + j] (-1: a panel that
    /// is no image of a sector panel), from the products of the group elements (reflections) and
    /// the images img of the sector panels.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn panel_images(&self) -> Vec<i32> {
        let (G, Ns, Nf) = (self.G as usize, self.Ns as usize, self.Nf as usize);
        let mut prod = vec![0usize; G * G];
        for a in 0..G {
            for b in 0..G {
                for c in 0..G {
                    let (ra, rb, rc) = (&self.refl[a], &self.refl[b], &self.refl[c]);
                    if rc.sx == ra.sx * rb.sx && rc.sy == ra.sy * rb.sy {
                        prod[a * G + b] = c;
                    }
                }
            }
        }
        let mut gj = vec![-1i32; G * Nf];
        for gi in 0..G {
            for g2 in 0..G {
                for n in 0..Ns {
                    gj[gi * Nf + self.img[g2 * Ns + n] as usize] = self.img[prod[gi * G + g2] * Ns + n];
                }
            }
        }
        gj
    }

    pub fn interior(&self, j: i32) -> bool {
        self.kind[j as usize] == SurfaceKind::Interior
    }
    pub fn press_z0(&self, j: i32) -> bool {
        self.kind[j as usize] == SurfaceKind::Press && self.ps[self.psurf[j as usize] as usize].z0
    }
    /// in z = 0
    pub fn on_surface(&self, j: i32) -> bool {
        let k = self.kind[j as usize];
        k == SurfaceKind::Interior || k == SurfaceKind::Lid || k == SurfaceKind::Plate || self.press_z0(j)
    }
    /// carries n_j
    pub fn body_surface(&self, j: i32) -> bool {
        let k = self.kind[j as usize];
        k == SurfaceKind::Wet || k == SurfaceKind::Sheet || k == SurfaceKind::Plate
    }
    /// a free-surface plate
    pub fn plate(&self, j: i32) -> bool {
        self.kind[j as usize] == SurfaceKind::Plate
    }
    /// a panel with the rows and columns of a thin sheet: a sheet, and at omega = inf a
    /// free-surface plate (theory §9.8: with its odd image the plate is a sheet in z = 0)
    pub fn sheet_like(&self, fc: &FreqCtx, j: i32) -> bool {
        let k = self.kind[j as usize];
        k == SurfaceKind::Sheet || (k == SurfaceKind::Plate && fc.mode == FreqMode::Infinite)
    }
    /// sheet rows at this frequency or limit: thin sheets, or free-surface plates at omega = inf
    pub fn sheet_rows(&self, fc: &FreqCtx) -> bool {
        self.sheets || (self.plates && fc.mode == FreqMode::Infinite)
    }
    pub fn free_damper(&self, j: i32) -> bool {
        let d = self.damper[j as usize];
        d >= 0 && self.dmp[d as usize].free
    }
    pub fn any_free_damper(&self) -> bool {
        for d in &self.dmp {
            if d.free {
                return true;
            }
        }
        false
    }
}

// ------------------------------------------------------------------ influence data

/// Frequency-independent parts for one field point against all panels:
/// Rankine source/dipole of the image sum sum_p 1/r_p (infinite depth
/// 1/r + 1/r', finite depth the six terms of green_fin.rs) and the
/// corrections of the log singularity relative to its centroid value (stored
/// in float32); for the omega = inf limit the sum with the free-surface images
/// negated. R = double in the diagnostic mode.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticEntry<R> {
    pub rs: R, // Rankine source, dipole; log corrections of source, dipole
    pub rd: R,
    pub dl: R,
    pub dd: R,
}

#[derive(Clone, Debug, Default)]
pub struct Static<R> {
    pub e: Vec<StaticEntry<R>>, // [field][Nf]
    pub inf: Vec<[R; 2]>,       // [field][Nf] source, dipole for omega = inf (if requested)
    // channel: the same entries at the field points reflected in the
    // walls, (x, W - y, z) and (x, -W - y, z)
    pub e1: Vec<StaticEntry<R>>,
    pub e2: Vec<StaticEntry<R>>,
    pub inf1: Vec<[R; 2]>,
    pub inf2: Vec<[R; 2]>,
    // sheet rows: sheet[k] the row of sector panel k (-1: none); per row
    // and panel the Rankine parts of n_x . grad of the dipole (h) and source (k)
    // integrals with the free-surface image added and, for omega = inf,
    // subtracted, and the log corrections of h and k
    pub sheet: Vec<i32>,
    pub sh: Vec<[R; 4]>,    // [row][Nf] h, k, log h, log k
    pub shinf: Vec<[R; 2]>, // [row][Nf] h, k for omega = inf (if requested)
    // channel: the same at the sheet points reflected in the walls, with their reflected normals
    pub sh1: Vec<[R; 4]>,
    pub sh2: Vec<[R; 4]>,
    pub shinf1: Vec<[R; 2]>,
    pub shinf2: Vec<[R; 2]>,
    // source formulation: [sector k][Nf] the Rankine gradient of the source
    // integral at the centroid of k (images included) and the log-correction gradient
    pub vg: Vec<[R; 3]>,
    pub vl: Vec<[R; 3]>,
    // the first field point with velocity rows (the rows of point q at (q - vfrom) Nf)
    pub vfrom: usize,
    // field points only: the rows are not stored but made for each point when it is evaluated
    // (when storing them would take more than FIELD_STATICS_LIMIT), with these parameters
    pub lazy: Option<LazyRows>,
    // channel: the same at the field points reflected in the walls
    pub vg1: Vec<[R; 3]>,
    pub vl1: Vec<[R; 3]>,
    pub vg2: Vec<[R; 3]>,
    pub vl2: Vec<[R; 3]>,
}

/// One frequency or limit: the kernel data shared by all pairs.
#[derive(Clone, Debug)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct FreqCtx {
    pub mode: FreqMode,
    pub w: f64, // radian frequency; K = w^2/g
    pub K: f64,
    pub k: f64, // wave number of the incident wave
    pub depth: f64,
    pub logc: f64,             // weight of the analytic log-singularity integral
    pub kappa: Vec<Complex64>, // lid condition phi_z = kappa phi per lid (theory §9.1)
    pub fd: FiniteDepth,
    pub ch: ChannelTail, // channel: propagating far-image tail (ordinary frequencies)
    pub Me: i32,         // channel: farthest image whose evanescent part is kept
}

impl Default for FreqCtx {
    fn default() -> Self {
        FreqCtx {
            mode: FreqMode::Ordinary,
            w: 0.0,
            K: 0.0,
            k: 0.0,
            depth: f64::INFINITY,
            logc: 0.0,
            kappa: Vec::new(),
            fd: FiniteDepth::default(),
            ch: ChannelTail::default(),
            Me: 0,
        }
    }
}

impl FreqCtx {
    pub fn finite(&self) -> bool {
        self.depth.is_finite()
    }
}

/// A view of static entries (the field point or one of its channel images).
#[derive(Clone, Copy, Debug)]
pub struct StaticView<'a, R> {
    pub e: &'a [StaticEntry<R>],
    pub inf: &'a [[R; 2]],
}

/// The parameters of the static rows of a field point made when it is evaluated (Static::lazy).
#[derive(Clone, Copy, Debug)]
pub struct LazyRows {
    pub logint: bool,
    pub tol: f64,
    pub depth: f64,
}

/// The bytes above which the static rows of the field points are made for each point when it is
/// evaluated instead of stored: the field evaluation then repeats them at every frequency, and the
/// memory stays that of one row per thread.
pub const FIELD_STATICS_LIMIT: f64 = 4.0 * 1024.0 * 1024.0 * 1024.0;

/// Whether the static rows of P field points over Nf panels (with velocity rows from vfrom on)
/// exceed FIELD_STATICS_LIMIT in the real type of `bytes_r` bytes; never in a channel (its wall
/// images keep the stored rows).
/// The environment variable HB_FIELD_STATICS_LIMIT (bytes) replaces the limit, read at every call
/// (once per solve) so that a test can set it in the process: tests/python/test_field.py sets 0 to
/// make the rows at evaluation. A value that is not a number refuses the solve before it starts
/// (env_hooks::environment_hooks); here it leaves the limit.
pub fn field_statics_lazy(points: usize, nf: usize, vrows: usize, bytes_r: usize, channel: bool) -> bool {
    let per = (4 * points + 6 * vrows) as f64 * nf as f64 * bytes_r as f64;
    let limit = crate::env_hooks::env_number::<f64>("HB_FIELD_STATICS_LIMIT")
        .ok()
        .flatten()
        .filter(|x| !x.is_nan())
        .unwrap_or(FIELD_STATICS_LIMIT);
    !channel && points > 0 && per > limit
}

/// The static entries of a field point that apply_kernel reads: a Static or a StaticView.
pub trait StaticRows<R> {
    fn e(&self) -> &[StaticEntry<R>];
    fn inf(&self) -> &[[R; 2]];
}

impl<R> StaticRows<R> for Static<R> {
    fn e(&self) -> &[StaticEntry<R>] {
        &self.e
    }
    fn inf(&self) -> &[[R; 2]] {
        &self.inf
    }
}

impl<R> StaticRows<R> for StaticView<'_, R> {
    fn e(&self) -> &[StaticEntry<R>] {
        self.e
    }
    fn inf(&self) -> &[[R; 2]] {
        self.inf
    }
}

/// Regular part of the kernel of a pair of points (green_fin.rs Kern;
/// symmetric in the two points, Wrz the derivative in the first one) and their horizontal distance R.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn pair_kernel(fc: &FreqCtx, x: &Vec3, xi: &Vec3, px: Option<&FinPoint>, pxi: Option<&FinPoint>) -> (Kern, f64) {
    let dx = x[0] - xi[0];
    let dy = x[1] - xi[1];
    let R = (dx * dx + dy * dy).sqrt();
    if fc.finite() {
        return (finite_kernel(&fc.fd, R, x[2], xi[2], px, pxi), R);
    }
    let mut k = Kern::default();
    if fc.mode != FreqMode::Ordinary {
        return (k, R); // 1/r +- 1/r' only
    }
    let K = fc.K;
    if R == 0.0 && x[2] == 0.0 && xi[2] == 0.0 {
        // coincident points on z = 0 (an interior panel and its own centroid):
        // Ft + 2 log(Y + R0) -> -2 (gamma - log 2), so K (Ft + 2 psi) with
        // psi = log((Y + R0)/K) tends to -2K (gamma - log 2 + log K)
        k.Wr = -2.0 * K * (GAMMA - LOG2 + K.ln());
        k.Wi = -2.0 * PI * K;
        return (k, R);
    }
    (wave_kern(K, &wave_kernel(K * R, -K * (x[2] + xi[2]))), R)
}

/// The kernels of the point x (fin_point data px in finite depth) and the centroids of the panels
/// js (fin_point data fpt by panel), as pair_kernel gives them (with the fin_point data in finite
/// depth), appended to out in the order of js: four at a time where wave_kernel_near4 or
/// wave_kernel_direct4 (infinite depth) or finite_kernel4 (finite depth) holds the four, else one
/// by one; equal to the bit.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn pair_kernels(
    fc: &FreqCtx,
    g: &Geometry,
    x: &Vec3,
    px: Option<&FinPoint>,
    fpt: &[FinPoint],
    js: &[i32],
    out: &mut Vec<(Kern, f64)>,
) {
    let fin = fc.finite();
    let four = if fin { px.is_some() } else { fc.mode == FreqMode::Ordinary };
    let K = fc.K;
    let mut e = 0;
    while e < js.len() {
        if four && e + 4 <= js.len() {
            let c: [&Vec3; 4] = std::array::from_fn(|k| &g.pg[js[e + k] as usize].c);
            let R: [f64; 4] = std::array::from_fn(|k| {
                let dx = x[0] - c[k][0];
                let dy = x[1] - c[k][1];
                (dx * dx + dy * dy).sqrt()
            });
            if fin {
                if fin4_applies(&fc.fd, &R) {
                    let zeta: [f64; 4] = std::array::from_fn(|k| c[k][2]);
                    let pzeta: [&FinPoint; 4] = std::array::from_fn(|k| &fpt[js[e + k] as usize]);
                    let w = finite_kernel4(&fc.fd, &R, x[2], &zeta, px.unwrap(), pzeta);
                    out.extend((0..4).map(|k| (w[k], R[k])));
                    e += 4;
                    continue;
                }
            } else {
                let Xs: [f64; 4] = std::array::from_fn(|k| K * R[k]);
                let Ys: [f64; 4] = std::array::from_fn(|k| -K * (x[2] + c[k][2]));
                let w = if near4_applies(&Xs, &Ys) {
                    Some(wave_kernel_near4(&Xs, &Ys))
                } else if direct4_applies(&Xs, &Ys) {
                    Some(wave_kernel_direct4(&Xs, &Ys))
                } else {
                    None
                };
                if let Some(w) = w {
                    out.extend((0..4).map(|k| (wave_kern(K, &w[k]), R[k])));
                    e += 4;
                    continue;
                }
            }
        }
        // one by one: the group of four (or the last points) does not apply
        let n = if four { (js.len() - e).min(4) } else { js.len() - e };
        for &j in &js[e..e + n] {
            let pj = if fin && px.is_some() { Some(&fpt[j as usize]) } else { None };
            out.push(pair_kernel(fc, x, &g.pg[j as usize].c, px, pj));
        }
        e += n;
    }
}

/// The kernel of a pair at an ordinary frequency in infinite depth from its wave kernel at
/// (X, Y) = (K R, -K (z + zeta)), as pair_kernel forms it.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn wave_kern(K: f64, w: &WaveKernel) -> Kern {
    let K2 = K * K;
    let mut k = Kern::default();
    // Y = -K (z + zeta): d/dzeta = d/dz = -K d/dY
    k.Wr = K * w.F;
    k.WrR = K2 * w.FX;
    k.WrZ = -K2 * w.FY;
    k.Wrz = k.WrZ;
    k.Wi = K * w.I;
    k.WiR = K2 * w.IX;
    k.WiZ = -K2 * w.IY;
    k.Wiz = k.WiZ;
    k
}

/// Complete influence of panel j at the field point x given the kernel of the
/// pair: (S, D), S = int G dS, D = int dG/dn dS. swapped: the kernel was evaluated with
/// x as its source point, so the source-depth derivative is Wrz.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn apply_kernel<R: Real, St: StaticRows<R>>(
    g: &Geometry,
    st: &St,
    row: usize,
    j: i32,
    x: &Vec3,
    fc: &FreqCtx,
    k: &Kern,
    R: f64,
    swapped: bool,
) -> (Complex64, Complex64) {
    let ju = j as usize;
    let p = &g.pg[ju];
    // singular part -2K psi, psi = log(r' + |z + zeta|) (theory §6.2), with a constant coefficient,
    // integrated analytically for every pair
    let c = fc.logc;
    let a = if R > 0.0 { ((x[0] - p.c[0]) * p.n[0] + (x[1] - p.c[1]) * p.n[1]) / R } else { 0.0 };
    let o = row + ju;
    let e = &st.e()[o];
    let rs = if fc.mode == FreqMode::Infinite { st.inf()[o][0].as_f64() } else { e.rs.as_f64() };
    let rd = if fc.mode == FreqMode::Infinite { st.inf()[o][1].as_f64() } else { e.rd.as_f64() };
    let A = p.area;
    let K = fc.K;
    let rz = if swapped { k.Wrz } else { k.WrZ };
    let iz = if swapped { k.Wiz } else { k.WiZ };
    let S = Complex64::new(rs + A * k.Wr - 2.0 * K * c * e.dl.as_f64(), A * k.Wi);
    let D = Complex64::new(
        rd + A * (-k.WrR * a + rz * p.n[2]) - 2.0 * K * c * e.dd.as_f64(),
        A * (-k.WiR * a + iz * p.n[2]),
    );
    (S, kind_column(g, fc, j, S, D))
}

/// The "dipole" column of panel j by its surface kind, from its source integral S and dipole
/// integral D: an interior free-surface panel K S at ordinary frequencies (dG/dzeta = K G on
/// zeta = 0, Lee, Newman & Zhu 1996 eq. 21), a pressure surface press_column, a lid lid_column,
/// a free-surface plate plate_column (at omega = inf its dipole D: the plate is then a sheet of
/// the odd-image problem, its coincident image doubling the dipole, theory §9.8), every other
/// panel D. The rule of apply_kernel and channel_influence; the sheet rows (lo_frequency.rs) take
/// the lid, plate and pressure rules only (their interior columns stay the normal derivative of
/// D), the Kochin functions the pressure rule only (no interior panels, no lids or plates there).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn kind_column(g: &Geometry, fc: &FreqCtx, j: i32, S: Complex64, D: Complex64) -> Complex64 {
    match g.kind[j as usize] {
        SurfaceKind::Interior if fc.mode == FreqMode::Ordinary => fc.K * S,
        SurfaceKind::Press => press_column(g, fc, j, S, D),
        SurfaceKind::Lid => lid_column(g, fc, j, S),
        SurfaceKind::Plate if fc.mode == FreqMode::Infinite => D,
        SurfaceKind::Plate => plate_column(fc, S),
        _ => D,
    }
}

/// The column of a free-surface plate panel: dG/dn = -dG/dzeta = -K G on zeta = 0 (the normal
/// -z into the water), so its dipole integral is -K S from its source integral (theory §9.7,
/// theory §9.8): the lid column of kappa = 0. At omega = 0 (K = 0) it vanishes, but the plate's
/// unknowns stay in use (its normal velocity enters the right-hand side).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn plate_column(fc: &FreqCtx, S: Complex64) -> Complex64 {
    if fc.mode == FreqMode::Ordinary { -fc.K * S } else { Complex64::new(0.0, 0.0) }
}

/// The column of a pressure surface j, -int phi (dG/dn + K G) dS: D + K S from its source and
/// dipole integrals; it vanishes in z = 0, where dG/dzeta = K G.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn press_column(g: &Geometry, fc: &FreqCtx, j: i32, S: Complex64, D: Complex64) -> Complex64 {
    if g.press_z0(j) { Complex64::new(0.0, 0.0) } else { D + fc.K * S }
}

/// The column of a lid panel j: the lid term -(K - kappa) int phi G dS of the integral equation
/// from its source integral S, stored as the "dipole" column (its coefficient enters with a
/// minus sign); 0 at the limits, where the lid unknowns are not used.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn lid_column(g: &Geometry, fc: &FreqCtx, j: i32, S: Complex64) -> Complex64 {
    if fc.mode == FreqMode::Ordinary {
        -(fc.K - fc.kappa[g.lid[j as usize] as usize]) * S
    } else {
        Complex64::new(0.0, 0.0)
    }
}

/// Wave part of a sheet row (infinite depth): centroid rule for
/// n_x . grad_x G_w (Kp) and n_x^T grad_x grad_xi G_w n_xi (H) of panel p at x,
/// G_w = K Ft(X, Y) + i K Gi(X, Y); the second derivatives of Ft follow from
/// Ft_Y = -Ft - 2/R0 and Laplace's equation Ft_XX + Ft_X/X + Ft_YY = 0, those of
/// Gi = -2 pi e^{-Y} J0(X) from Bessel's equation. Returns (H, Kp).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn sheet_wave(fc: &FreqCtx, x: &Vec3, nx: &Vec3, p: &PanelGeom) -> (Complex64, Complex64) {
    if fc.mode != FreqMode::Ordinary || fc.finite() {
        return (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    }
    let K = fc.K;
    let K2 = K * K;
    let K3 = K2 * K;
    let dx = x[0] - p.c[0];
    let dy = x[1] - p.c[1];
    let X = K * (dx * dx + dy * dy).sqrt();
    let Y = -K * (x[2] + p.c[2]);
    let w = wave_kernel(X, Y);
    let R0 = (X * X + Y * Y).sqrt();
    let iR03 = 1.0 / (R0 * R0 * R0);
    let FYY = -w.FY + 2.0 * Y * iR03;
    let FXY = -w.FX + 2.0 * X * iR03;
    let (FX_X, IX_X) = wave_x_over_x(X, Y, &w, FYY); // the quotients near the axis as the tables allow
    let FXX = -FX_X - FYY;
    let IXX = -w.I - IX_X;
    let IXY = -w.IX;
    let IYY = w.I;
    let (kr, hr) = pair_derivatives(x, &p.c, nx, &p.n, K2 * w.FX, K2 * w.FY, K3 * FXX, K3 * FX_X, K3 * FXY, K3 * FYY);
    let (ki, hi) = pair_derivatives(x, &p.c, nx, &p.n, K2 * w.IX, K2 * w.IY, K3 * IXX, K3 * IX_X, K3 * IXY, K3 * IYY);
    (p.area * Complex64::new(hr, hi), p.area * Complex64::new(kr, ki))
}

/// log|2 sinh(f z) / z| (f = pi / 2W, z = a + i s) and its derivatives in a and
/// s: the sum over one channel image family of log of the horizontal distance
/// without its j = 0 term, up to a constant. Smooth at z = 0 (log 2f +
/// Re (fz)^2 / 6); for Re q >= 0, log|2 sinh q| = Re q + log|1 - e^{-2q}| and
/// coth q = (1 + e^{-2q}) / (1 - e^{-2q}), so long channels do not overflow.
/// d/da log|g| = Re(g'/g), d/ds = -Im(g'/g) for g analytic in z. Returns (v, dv/da, dv/ds).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn periodic_log_regular(a: f64, s: f64, W: f64) -> (f64, f64, f64) {
    let f = PI / (2.0 * W);
    let z = Complex64::new(a, s);
    let qz = f * z;
    let v: f64;
    let d: Complex64; // g'/g of g = 2 sinh(f z) / z
    if qz.norm() < 1e-3 {
        let q2 = qz * qz;
        v = (2.0 * f).ln() + (q2 / 6.0 - q2 * q2 / 180.0).re;
        d = f * (qz / 3.0 - q2 * qz / 45.0);
    } else {
        let sg = if a >= 0.0 { 1.0 } else { -1.0 }; // log|2 sinh q| is even in q
        let q = sg * qz;
        let e = (-2.0 * q).exp();
        v = q.re + (1.0 - e).norm().ln() - z.norm().ln();
        d = sg * f * (1.0 + e) / (1.0 - e) - 1.0 / z;
    }
    (v, d.re, -d.im)
}

/// Regular part of a sheet row in finite depth, the centroid rule:
/// Kp = n_x . grad_x G_reg and H = n_x^T grad_x grad_xi G_reg n_xi from the
/// analytic first and second derivatives of the finite-depth kernel
/// (finite_kernel_hess), with grad_x grad_xi of f(R, z, zeta):
///   horizontal-horizontal -(f_RR e e^T + f_R/R (I - e e^T)), x_h-zeta f_Rzeta e,
///   z-xi_h -f_Rz e^T, z-zeta f_zzeta (e the horizontal unit vector from xi to x). Returns (H, Kp).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn sheet_wave_fd(
    fc: &FreqCtx,
    x: &Vec3,
    nx: &Vec3,
    p: &PanelGeom,
    px: Option<&FinPoint>,
    pp: Option<&FinPoint>,
) -> (Complex64, Complex64) {
    let dx = x[0] - p.c[0];
    let dy = x[1] - p.c[1];
    let R = (dx * dx + dy * dy).sqrt();
    let (k, s) = finite_kernel_hess(&fc.fd, R, x[2], p.c[2], px, pp);
    let ex = if R > 0.0 { dx / R } else { 0.0 };
    let ey = if R > 0.0 { dy / R } else { 0.0 };
    let ax = nx[0] * ex + nx[1] * ey; // n . e
    let an = p.n[0] * ex + p.n[1] * ey;
    let hh = nx[0] * p.n[0] + nx[1] * p.n[1];
    let second = |RR: f64, R_R: f64, RZ: f64, Rz: f64, zZ: f64| -> f64 {
        -(RR * ax * an + R_R * (hh - ax * an)) + ax * RZ * p.n[2] - nx[2] * Rz * an + nx[2] * zZ * p.n[2]
    };
    let H = p.area * Complex64::new(second(s.RR, s.R_R, s.RZ, s.Rz, s.zZ), second(s.iRR, s.iR_R, s.iRZ, s.iRz, s.izZ));
    let Kp = p.area * Complex64::new(k.WrR * ax + k.Wrz * nx[2], k.WiR * ax + k.Wiz * nx[2]);
    (H, Kp)
}

fn sgn(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The depth factor E(z, zeta) of the propagating tail of the channel kernel and its derivatives
/// from the one-point factors of the field point (px) and of the source point (pj), with
/// exp(-k0 v_p) = f_p(z) g_p(zeta) (green_fin.rs propagating_products, propagating_factor).
/// Returns (E, dE/dz, dE/dzeta, d2E/dz dzeta).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn tail_depth_factor(k0: f64, px: &FinPoint, pj: &FinPoint) -> (f64, f64, f64, f64) {
    let [E, EZ, Ez, EzZ] = propagating_factor(k0, propagating_products(px, pj));
    (E, Ez, EZ, EzZ)
}

/// The ordinate of the field point y in the channel image m (period 2W, the odd images
/// reflected): y - m W for even m, m W - y for odd m. Returns (y_m, dy_m/dy).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn channel_image_y(m: i32, y: f64, W: f64) -> (f64, f64) {
    if m % 2 == 0 { (y - m as f64 * W, 1.0) } else { (m as f64 * W - y, -1.0) }
}

/// Channel influence of panel j at x: the body and the images in the two
/// walls with the panel integrals of the static data, the evanescent series of
/// the images 2 <= |m| <= Me and the propagating tail
/// -i A0 E [Q_even(|dx|, |y - eta|) + Q_odd(|dx|, |y + eta|)] by the centroid rule
/// (the regular part of the low-order method). Returns (S, D).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn channel_influence<R: Real>(
    g: &Geometry,
    st: &Static<R>,
    row: usize,
    j: i32,
    x: &Vec3,
    px: &FinPoint,
    pj: &FinPoint,
    fc: &FreqCtx,
) -> (Complex64, Complex64) {
    let W = g.W;
    let p = &g.pg[j as usize];
    let xs: [Vec3; 3] = [*x, v3(x[0], W - x[1], x[2]), v3(x[0], -W - x[1], x[2])];
    let views: [StaticView<R>; 3] = [
        StaticView { e: &st.e, inf: &st.inf },
        StaticView { e: &st.e1, inf: &st.inf1 },
        StaticView { e: &st.e2, inf: &st.inf2 },
    ];
    let mut S = Complex64::new(0.0, 0.0);
    let mut D = Complex64::new(0.0, 0.0);
    for m in 0..3 {
        let (k, Rh) = pair_kernel(fc, &xs[m], &p.c, Some(px), Some(pj));
        let (s, d) = apply_kernel(g, &views[m], row, j, &xs[m], fc, &k, Rh, false);
        let f = if m == 0 { 1.0 } else { g.Rc }; // the images scaled by the reflection factor
        S += f * s;
        D += f * d;
        if g.Rc == 0.0 {
            // no reflection: the body alone in the open sea of the same depth, no wall image
            // (frequency_context makes neither the far images nor the tail)
            return (S, D);
        }
    }
    let (mut Se, mut De) = channel_far(fc, W, x, &p.c, &p.n, p.area, px, pj);
    Se *= g.Rc;
    De *= g.Rc;
    // the column of each kind as in apply_kernel; on interior panels every far term satisfies
    // the free-surface condition dG/dzeta = K G on zeta = 0, exactly rather than from the
    // tabulated derivatives
    De = kind_column(g, fc, j, Se, De);
    S += Se;
    D += De;
    (S, D)
}

/// The far images of the channel kernel (the images 2 <= |m| <= Me by the evanescent series, at
/// omega = 0 the closed-form periodic log term, at ordinary frequencies the propagating tail)
/// of a source point xi with normal n and weight w at the field point x: (w G_far, w dG_far/dn),
/// without the reflection factor. The point rule of the low-order panels (w the area) and of the
/// higher-order quadrature points (w the point weight).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn channel_far(
    fc: &FreqCtx,
    W: f64,
    x: &Vec3,
    xi: &Vec3,
    n: &Vec3,
    w: f64,
    px: &FinPoint,
    pj: &FinPoint,
) -> (Complex64, Complex64) {
    let mut Se = Complex64::new(0.0, 0.0);
    let mut De = Complex64::new(0.0, 0.0);
    for am in 2..=fc.Me {
        for sg in [-1, 1] {
            let (ym, _) = channel_image_y(sg * am, x[1], W);
            let dx = x[0] - xi[0];
            let dy = ym - xi[1];
            let Rm = (dx * dx + dy * dy).sqrt();
            let (gv, gR, gz) = finite_series(&fc.fd, Rm, px, pj);
            Se += w * gv;
            De += w * (-gR * (dx * n[0] + dy * n[1]) / Rm + gz * n[2]);
        }
    }
    if fc.mode == FreqMode::Zero {
        // omega = 0: the 2D term -(2/h) log R of the finite-depth kernel for
        // the images |m| >= 2, both families of period 2W summed in closed form,
        // sum_j log|z - 2jWi| = log|2 sinh(pi z / 2W)| + const (z = dx + i s), minus
        // the images already included (m = 0: family A, j = 0; m = +-1: family B,
        // j = 0, -1). The constant cancels for modes without net flux (rule 9).
        let dx = x[0] - xi[0];
        let sA = x[1] - xi[1];
        let sB = x[1] + xi[1] - W;
        let (vA, vAx, vAs) = periodic_log_regular(dx, sA, W);
        let (mut vB, mut vBx, mut vBs) = periodic_log_regular(dx, sB, W);
        // family B: also without its j = -1 term (the image -eta - W, included as m = -1)
        let s2 = sB + 2.0 * W;
        let r2 = dx * dx + s2 * s2;
        vB -= 0.5 * r2.ln();
        vBx -= dx / r2;
        vBs -= s2 / r2;
        let c = -2.0 / fc.depth;
        // source derivatives: d/dxi = -d/dx; d/deta = -d/ds_A, +d/ds_B
        Se += w * c * (vA + vB);
        De += w * c * (-(vAx + vBx) * n[0] + (-vAs + vBs) * n[1]);
    }
    if fc.mode == FreqMode::Ordinary {
        let dx = x[0] - xi[0];
        let u1 = x[1] - xi[1];
        let u2 = x[1] + xi[1];
        let (q1, q1x, q1u) = fc.ch.eval(false, dx.abs(), u1.abs());
        let (q2, q2x, q2u) = fc.ch.eval(true, dx.abs(), u2.abs());
        let (E, _, EZ, _) = tail_depth_factor(fc.fd.k0, px, pj);
        let c = Complex64::new(0.0, -fc.fd.A0);
        let Q = q1 + q2;
        let gx = c * E * (-sgn(dx) * (q1x + q2x));
        let gy = c * E * (-sgn(u1) * q1u + sgn(u2) * q2u);
        Se += w * c * E * Q;
        De += w * (gx * n[0] + gy * n[1] + c * EZ * Q * n[2]);
    }
    (Se, De)
}

/// The second derivatives of periodic_log_regular in a and s: v = Re L(z) with L analytic, so
/// v_aa = Re L'', v_as = -Im L'', v_ss = -Re L''; L'' = (g'/g)' = -f^2 csch^2(f z) + 1/z^2,
/// csch^2 q = 4 e^{-2q} / (1 - e^{-2q})^2 (near z = 0 f^2/3 - f^4 z^2/15). Returns (v_aa, v_as, v_ss).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn periodic_log_hess(a: f64, s: f64, W: f64) -> (f64, f64, f64) {
    let f = PI / (2.0 * W);
    let z = Complex64::new(a, s);
    let qz = f * z;
    let d2 = if qz.norm() < 1e-3 {
        f * f / 3.0 - f.powi(4) * z * z / 15.0
    } else {
        let q = if a >= 0.0 { qz } else { -qz };
        let e = (-2.0 * q).exp();
        -f * f * 4.0 * e / ((1.0 - e) * (1.0 - e)) + 1.0 / (z * z)
    };
    (d2.re, -d2.im, -d2.re)
}

/// The far images of the channel kernel in a sheet row (the terms of channel_far): of a source
/// point xi with normal n and weight w at the sheet point x with normal nx, (w nx^T grad_x grad_xi
/// G_far n, w nx . grad_x G_far), without the reflection factor. The second derivatives of the
/// evanescent series (finite_series_hess, the horizontal-horizontal form of sheet_wave_fd with
/// the y derivative reversed for the odd images), of the periodic log at omega = 0
/// (periodic_log_hess) and of the tail polynomials (ChannelTail::eval2). Returns (H, Kp).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn channel_far_sheet(
    fc: &FreqCtx,
    W: f64,
    x: &Vec3,
    nx: &Vec3,
    xi: &Vec3,
    n: &Vec3,
    w: f64,
    px: &FinPoint,
    pj: &FinPoint,
) -> (Complex64, Complex64) {
    let mut H = Complex64::new(0.0, 0.0);
    let mut Kp = Complex64::new(0.0, 0.0);
    let dx = x[0] - xi[0];
    for am in 2..=fc.Me {
        for sg in [-1, 1] {
            let (ym, s) = channel_image_y(sg * am, x[1], W); // s = dym/dy
            let dy = ym - xi[1];
            let Rm = (dx * dx + dy * dy).sqrt();
            let [fR, fz, fRR, fRz, fRZ, fzZ] = finite_series_hess(&fc.fd, Rm, px, pj);
            let (ex, ey) = (dx / Rm, dy / Rm);
            let ax = nx[0] * ex + s * nx[1] * ey;
            let an = n[0] * ex + n[1] * ey;
            let hh = nx[0] * n[0] + s * nx[1] * n[1];
            H += w
                * (-(fRR * ax * an + fR / Rm * (hh - ax * an)) + ax * fRZ * n[2] - nx[2] * fRz * an
                    + nx[2] * fzZ * n[2]);
            Kp += w * (fR * ax + fz * nx[2]);
        }
    }
    if fc.mode == FreqMode::Zero {
        // the periodic log of channel_far: family A in (dx, y - eta), B in (dx, y + eta - W)
        // without its j = -1 term log|dx + i (y + eta + W)|
        let sA = x[1] - xi[1];
        let sB = x[1] + xi[1] - W;
        let (_, vAx, vAs) = periodic_log_regular(dx, sA, W);
        let (_, mut vBx, mut vBs) = periodic_log_regular(dx, sB, W);
        let (aA, bA, cA) = periodic_log_hess(dx, sA, W);
        let (mut aB, mut bB, mut cB) = periodic_log_hess(dx, sB, W);
        let z2 = Complex64::new(dx, sB + 2.0 * W);
        let l1 = 1.0 / z2;
        let l2 = -1.0 / (z2 * z2);
        vBx -= l1.re;
        vBs += l1.im;
        aB -= l2.re;
        bB += l2.im;
        cB += l2.re;
        let c = -2.0 / fc.depth;
        // grad_x grad_xi: A -[[aa, as], [as, ss]]; B [[-aa, as], [-as, ss]]
        let hA = -(nx[0] * (aA * n[0] + bA * n[1]) + nx[1] * (bA * n[0] + cA * n[1]));
        let hB = nx[0] * (-aB * n[0] + bB * n[1]) + nx[1] * (-bB * n[0] + cB * n[1]);
        H += w * c * (hA + hB);
        Kp += w * c * (nx[0] * (vAx + vBx) + nx[1] * (vAs + vBs));
    }
    if fc.mode == FreqMode::Ordinary {
        let u1 = x[1] - xi[1];
        let u2 = x[1] + xi[1];
        let (q1, q1x, q1u) = fc.ch.eval(false, dx.abs(), u1.abs());
        let (q2, q2x, q2u) = fc.ch.eval(true, dx.abs(), u2.abs());
        let (q1xx, q1xu, q1uu) = fc.ch.eval2(false, dx.abs(), u1.abs());
        let (q2xx, q2xu, q2uu) = fc.ch.eval2(true, dx.abs(), u2.abs());
        let (sx, s1, s2) = (sgn(dx), sgn(u1), sgn(u2));
        let (E, Ez, EZ, EzZ) = tail_depth_factor(fc.fd.k0, px, pj); // Ez = dE/dz, EZ = dE/dzeta
        let Q = q1 + q2;
        // the derivatives of Q in x (field) and xi (source), and the mixed ones
        let Qx = [sx * (q1x + q2x), s1 * q1u + s2 * q2u];
        let Qs = [-sx * (q1x + q2x), -s1 * q1u + s2 * q2u];
        let Qxs = [[-(q1xx + q2xx), sx * (-s1 * q1xu + s2 * q2xu)], [-sx * (s1 * q1xu + s2 * q2xu), -q1uu + q2uu]];
        let c = Complex64::new(0.0, -fc.fd.A0);
        let mut hv = EzZ * Q * nx[2] * n[2];
        for a in 0..2 {
            hv += EZ * Qx[a] * nx[a] * n[2] + Ez * Qs[a] * nx[2] * n[a];
            for b in 0..2 {
                hv += E * Qxs[a][b] * nx[a] * n[b];
            }
        }
        H += w * c * hv;
        Kp += w * c * (E * (Qx[0] * nx[0] + Qx[1] * nx[1]) + Ez * Q * nx[2]);
    }
    (H, Kp)
}

/// grad_x of the channel source integral of panel j at x, the field
/// derivatives of the terms of channel_influence: the three near images at the
/// field points reflected in the walls (Rankine gradients and log corrections
/// from the static rows `vg[m]`, `vl[m]` at offset o, the wave part at the
/// centroid; d/dy = -d/dy' for the reflected points), the series images
/// 2 <= |m| <= Me and the propagating tail. Ordinary frequencies.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn channel_grad<R: Real>(
    g: &Geometry,
    vg: [&[[R; 3]]; 3],
    vl: [&[[R; 3]]; 3],
    o: usize,
    j: i32,
    x: &Vec3,
    px: &FinPoint,
    pj: &FinPoint,
    fc: &FreqCtx,
) -> [Complex64; 3] {
    let W = g.W;
    let p = &g.pg[j as usize];
    let A = p.area;
    let c2 = 2.0 * fc.K * fc.logc;
    let xs: [Vec3; 3] = [*x, v3(x[0], W - x[1], x[2]), v3(x[0], -W - x[1], x[2])];
    let mut v = [Complex64::new(0.0, 0.0); 3];
    for m in 0..3 {
        let (k, Rh) = pair_kernel(fc, &xs[m], &p.c, Some(px), Some(pj));
        let ax = if Rh > 0.0 { (xs[m][0] - p.c[0]) / Rh } else { 0.0 };
        let ay = if Rh > 0.0 { (xs[m][1] - p.c[1]) / Rh } else { 0.0 };
        let wr = [k.WrR * ax, k.WrR * ay, k.Wrz];
        let wi = [k.WiR * ax, k.WiR * ay, k.Wiz];
        let sy = [1.0, if m == 0 { 1.0 } else { -1.0 }, 1.0];
        let f = if m == 0 { 1.0 } else { g.Rc };
        for d in 0..3 {
            v[d] += f * sy[d] * Complex64::new(vg[m][o][d].as_f64() + A * wr[d] - c2 * vl[m][o][d].as_f64(), A * wi[d]);
        }
        if g.Rc == 0.0 {
            return v; // no reflection: no wall image (as channel_influence)
        }
    }
    let mut far = [Complex64::new(0.0, 0.0); 3];
    for am in 2..=fc.Me {
        for sg in [-1, 1] {
            let (ym, s) = channel_image_y(sg * am, x[1], W); // s = dym/dy
            let dx = x[0] - p.c[0];
            let dy = ym - p.c[1];
            let Rm = (dx * dx + dy * dy).sqrt();
            let (_, gR, gz) = finite_series(&fc.fd, Rm, pj, px); // the depths exchanged: gz = d/dz
            far[0] += A * gR * dx / Rm;
            far[1] += A * gR * dy / Rm * s;
            far[2] += A * gz;
        }
    }
    if fc.mode == FreqMode::Ordinary {
        let dx = x[0] - p.c[0];
        let u1 = x[1] - p.c[1];
        let u2 = x[1] + p.c[1];
        let (q1, q1x, q1u) = fc.ch.eval(false, dx.abs(), u1.abs());
        let (q2, q2x, q2u) = fc.ch.eval(true, dx.abs(), u2.abs());
        let (E, Ez, _, _) = tail_depth_factor(fc.fd.k0, px, pj); // Ez = dE/dz
        let c = Complex64::new(0.0, -fc.fd.A0);
        far[0] += A * c * E * (sgn(dx) * (q1x + q2x));
        far[1] += A * c * E * (sgn(u1) * q1u + sgn(u2) * q2u);
        far[2] += A * c * Ez * (q1 + q2);
    }
    for d in 0..3 {
        v[d] += g.Rc * far[d];
    }
    v
}

/// The influence (S, D) of panel j at x (pair_kernel and apply_kernel).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn influence<R: Real, St: StaticRows<R>>(
    g: &Geometry,
    st: &St,
    row: usize,
    j: i32,
    x: &Vec3,
    fc: &FreqCtx,
) -> (Complex64, Complex64) {
    let (k, R) = pair_kernel(fc, x, &g.pg[j as usize].c, None, None);
    apply_kernel(g, st, row, j, x, fc, &k, R, false)
}

// ------------------------------------------------------------------ problem data

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Column {
    pub coord: i32, // zero-based global coordinate
    pub body: i32,
    pub mode: i32, // 1..6 for a rigid mode, 0 for a pressure mode, -1 for a generalized mode
    pub ps: i32,   // pressure surface of a pressure mode (index into Geometry::ps)
    pub shape: PressureShape,
    pub gm: i32, // a generalized mode: its index in the body-major list of all generalized modes
}

impl Default for Column {
    fn default() -> Self {
        Column { coord: 0, body: 0, mode: 0, ps: -1, shape: PressureShape::Uniform, gm: -1 }
    }
}

// field points inside tanks: from their tank problems, [F][K] and [F][K][3] per point
#[derive(Clone, Debug, Default)]
pub struct TankField {
    pub point: i32,
    pub rad: Vec<Complex32>,
    pub vel: Vec<Complex32>,
}

// the pressure drift of the tank fluid, low order: per tank its wall points (body-local,
// normals into the tank fluid, the height below the tank free surface), its waterline (panel
// edges in the tank free surface, the point of the panel) and per ordinary frequency the
// potential and velocity (body axes) of every body column per unit velocity, [F][P][K] and
// [F][P][K][3]
#[derive(Clone, Debug, Default)]
pub struct TankDrift {
    pub body: i32,
    pub rho: f64,
    pub x: Vec<Vec3>,
    pub n: Vec<Vec3>,
    pub area: Vec<f64>,
    pub zrel: Vec<f64>,
    pub wl_pt: Vec<i32>,
    pub wl_x: Vec<Vec3>,
    pub wl_n: Vec<Vec3>,
    pub wl_dl: Vec<f64>,
    pub phi: Vec<Complex64>,
    pub vel: Vec<Complex64>,
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct Plan {
    pub rad: Vec<Column>, // radiation columns
    pub omegas: Vec<f64>, // ordinary, input order
    pub heads: Vec<f64>,  // radians
    pub D: i32,
    pub diffraction: bool,
    pub radiation: bool,
    pub rao: bool,
    pub logint: bool,
    pub total: bool,
    pub rho: f64,
    pub grav: f64,
    pub coords: Vec<Column>, // every coordinate (D), as rad
    pub MM: Vec<f64>,        // D x D motion matrices with the pressure-coordinate terms
    pub CC: Vec<f64>,
    pub BB: Vec<f64>,
    pub NK: i32,           // Kochin angles 2 pi t / NK (mean drift, theory §11.1)
    pub pdrift: bool,      // mean drift by pressure integration (theory §11.2)
    pub pqtf: QtfHeadings, // its quadratic QTFs
    pub cqtf: bool,        // the control-surface QTFs too, in the headings of pqtf
    pub cdrift: bool,      // mean drift by the control surfaces
    pub fvel: bool,        // velocities at the field points (theory §4.1)
    pub bpres: bool,       // pressure and velocity at the body panels
    pub bvel: bool,
    pub bp: Vec<i32>, // the body panels of these outputs (wetted, not wall images)
    // internal tanks: their added mass and damping in the body coordinates, [F][D][K], [D][K]
    pub tA: Vec<f64>,
    pub tB: Vec<f64>,
    pub tA0: Vec<f64>,
    pub tAinf: Vec<f64>,
    pub tfield: Vec<TankField>,
    pub tdrift: Vec<TankDrift>,
}

impl Default for Plan {
    fn default() -> Self {
        Plan {
            rad: Vec::new(),
            omegas: Vec::new(),
            heads: Vec::new(),
            D: 0,
            diffraction: false,
            radiation: false,
            rao: false,
            logint: true,
            total: false,
            rho: 0.0,
            grav: 0.0,
            coords: Vec::new(),
            MM: Vec::new(),
            CC: Vec::new(),
            BB: Vec::new(),
            NK: 0,
            pdrift: false,
            pqtf: QtfHeadings::None,
            cqtf: false,
            cdrift: false,
            fvel: false,
            bpres: false,
            bvel: false,
            bp: Vec::new(),
            tA: Vec::new(),
            tB: Vec::new(),
            tA0: Vec::new(),
            tAinf: Vec::new(),
            tfield: Vec::new(),
            tdrift: Vec::new(),
        }
    }
}

/// Far-field (Kochin) kernel of the direction theta (theory §11.1)
/// at a point x with normal n in the frame, (c, s) = (cos, sin) of theta
/// in the frame: g = Z(zeta) exp(i k (x c + y s)), its normal derivative and the
/// theta derivatives of both.
#[derive(Clone, Copy, Debug, Default)]
pub struct FarKern {
    pub g: Complex64,
    pub gn: Complex64,
    pub gt: Complex64,
    pub gnt: Complex64,
}

/// The depth factor of the wave k = fc.k at the depth z and its z derivative (Z, dZ/dz):
/// Z = cosh k(z + h) / cosh kh = (e^{kz} + e^{-k(z + 2h)}) / (1 + e^{-2kh}) (e^{kz} for h = inf).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn depth_factor(fc: &FreqCtx, z: f64) -> (f64, f64) {
    let k = fc.k;
    let mut Z = (k * z).exp();
    let mut Zz = k * Z;
    if fc.finite() {
        let h = fc.depth;
        let b = (-k * (z + 2.0 * h)).exp();
        let d = 1.0 + (-2.0 * k * h).exp();
        Zz = k * (Z - b) / d;
        Z = (Z + b) / d;
    }
    (Z, Zz)
}

#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn far_kernel(fc: &FreqCtx, x: &Vec3, n: &Vec3, c: f64, s: f64) -> FarKern {
    let k = fc.k;
    let (Z, Zz) = depth_factor(fc, x[2]);
    let ph = k * (x[0] * c + x[1] * s);
    let e = Complex64::new(ph.cos(), ph.sin());
    let ik = Complex64::new(0.0, k);
    let dt = -x[0] * s + x[1] * c;
    let dnt = -n[0] * s + n[1] * c;
    let mut f = FarKern::default();
    f.g = Z * e;
    f.gn = e * (Z * ik * (n[0] * c + n[1] * s) + Zz * n[2]);
    f.gt = f.g * ik * dt;
    f.gnt = f.gn * ik * dt + e * Z * ik * dnt;
    f
}

/// A Kochin value H of the frame (direction theta - yaw) and its derivative Ht in global axes:
/// the factor exp(i k (px cos + py sin)) of the frame origin (px, py), and its derivative.
/// Returns (H, Ht) in global axes.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn kochin_global(
    fc: &FreqCtx,
    px: f64,
    py: f64,
    theta: f64,
    H: Complex64,
    Ht: Complex64,
) -> (Complex64, Complex64) {
    let c = theta.cos();
    let s = theta.sin();
    let ph = fc.k * (px * c + py * s);
    let e = Complex64::new(ph.cos(), ph.sin());
    let Ht = e * (Ht + Complex64::new(0.0, fc.k * (-px * s + py * c)) * H);
    (e * H, Ht)
}

/// Incident wave in the frame (unit amplitude): (value, gradient) at x, with the depth factor Z
/// of depth_factor.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn incident_one(g: &Geometry, pl: &Plan, fc: &FreqCtx, beta: f64, x: &Vec3) -> (Complex64, [Complex64; 3]) {
    let k = fc.k;
    let w = fc.w;
    let cb = (beta - g.yaw * PI / 180.0).cos();
    let sb = (beta - g.yaw * PI / 180.0).sin();
    let phase = -k * (g.px * beta.cos() + g.py * beta.sin()) - k * (x[0] * cb + x[1] * sb);
    let (Z, Zz) = depth_factor(fc, x[2]);
    let e = Complex64::new(0.0, pl.grav / w) * Complex64::new(phase.cos(), phase.sin());
    let phi = e * Z;
    (phi, [phi * Complex64::new(0.0, -k * cb), phi * Complex64::new(0.0, -k * sb), e * Zz])
}

/// With walls the incident wave includes its reflections (theory §1.2): the sum of
/// the wave at the reflected points (the global frame is used with walls). Returns (value, gradient).
#[inline]
pub fn incident(g: &Geometry, pl: &Plan, fc: &FreqCtx, beta: f64, x: &Vec3) -> (Complex64, [Complex64; 3]) {
    let z = Complex64::new(0.0, 0.0);
    let mut phi = z;
    let mut grad = [z; 3];
    for w in &g.walls {
        let (p, gr) = incident_one(g, pl, fc, beta, &v3(w.sx * x[0], w.sy * x[1], x[2]));
        phi += p;
        grad[0] += w.sx * gr[0];
        grad[1] += w.sy * gr[1];
        grad[2] += gr[2];
    }
    (phi, grad)
}

#[derive(Clone, Debug, Default)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct Output {
    pub A: Vec<f32>, // [F][D][K]
    pub B: Vec<f32>,
    pub X: Vec<Complex32>, // [F][H][D]
    pub FK: Vec<Complex32>,
    pub SC: Vec<Complex32>,
    pub M: Vec<Complex32>,
    pub HK: Vec<Complex32>,   // [F][K][H]
    pub frad: Vec<Complex32>, // [F][P][K], [F][P][H]
    pub finc: Vec<Complex32>,
    pub fsca: Vec<Complex32>,
    pub ftot: Vec<Complex32>,
    pub A0: Vec<f32>, // [D][K] limits
    pub Ainf: Vec<f32>,
    pub lrad: Vec<Complex32>, // [F][L][K], [F][L][H] lid elevations
    pub ldif: Vec<Complex32>,
    pub prad: Vec<Complex32>, // [F][Q][K], [F][Q][H] pressure-surface elevations
    pub pdif: Vec<Complex32>,
    pub kr: Vec<Complex32>, // [F][NK][K], [F][NK][H] Kochin functions and d/dtheta
    pub krt: Vec<Complex32>,
    pub kd: Vec<Complex32>,
    pub kdt: Vec<Complex32>,
    pub PD: Vec<f64>,         // [F][H][9] mean drift by pressure integration
    pub PG: Vec<f64>,         // [F][H][G] its generalized forces (G the generalized modes of all bodies)
    pub CD: Vec<f64>,         // [F][H][9] mean drift by the control surfaces
    pub qtf: Vec<QtfSamples>, // [F] the first-order values of the quadratic QTFs (pressure drift)
    pub fvr: Vec<Complex32>,  // [F][P][K][3], [F][P][H][3] field velocities (global axes)
    pub fvi: Vec<Complex32>,
    pub fvs: Vec<Complex32>,
    pub fvt: Vec<Complex32>,
    pub bpr: Vec<Complex32>, // [F][Nb][K], [F][Nb][H] and [..][3]
    pub bpd: Vec<Complex32>,
    pub bvr: Vec<Complex32>,
    pub bvd: Vec<Complex32>,
}

// The outputs of one frequency (one) into those of all (all) at frequency fi.
fn put_at<T: Copy>(all: &mut [T], fi: usize, one: &[T]) {
    let n = one.len();
    all[fi * n..(fi + 1) * n].copy_from_slice(one);
}

impl Output {
    /// The outputs of the ordinary frequency fi, computed as the outputs of one frequency
    /// (solve_driver::allocate_outputs with F = 1), written at fi.
    pub fn put_frequency(&mut self, fi: usize, one: &Output) {
        put_at(&mut self.A, fi, &one.A);
        put_at(&mut self.B, fi, &one.B);
        for (all, o) in [
            (&mut self.X, &one.X),
            (&mut self.FK, &one.FK),
            (&mut self.SC, &one.SC),
            (&mut self.M, &one.M),
            (&mut self.HK, &one.HK),
            (&mut self.frad, &one.frad),
            (&mut self.finc, &one.finc),
            (&mut self.fsca, &one.fsca),
            (&mut self.ftot, &one.ftot),
            (&mut self.lrad, &one.lrad),
            (&mut self.ldif, &one.ldif),
            (&mut self.prad, &one.prad),
            (&mut self.pdif, &one.pdif),
            (&mut self.kr, &one.kr),
            (&mut self.krt, &one.krt),
            (&mut self.kd, &one.kd),
            (&mut self.kdt, &one.kdt),
            (&mut self.fvr, &one.fvr),
            (&mut self.fvi, &one.fvi),
            (&mut self.fvs, &one.fvs),
            (&mut self.fvt, &one.fvt),
            (&mut self.bpr, &one.bpr),
            (&mut self.bpd, &one.bpd),
            (&mut self.bvr, &one.bvr),
            (&mut self.bvd, &one.bvd),
        ] {
            put_at(all, fi, o);
        }
        put_at(&mut self.PD, fi, &one.PD);
        put_at(&mut self.PG, fi, &one.PG);
        put_at(&mut self.CD, fi, &one.CD);
        if let Some(s) = one.qtf.first() {
            self.qtf[fi] = s.clone();
        }
    }

    /// The added mass of a limit (its outputs as those of one frequency).
    pub fn put_limit(&mut self, mode: FreqMode, one: &Output) {
        if mode == FreqMode::Zero {
            self.A0.copy_from_slice(&one.A0);
        } else {
            self.Ainf.copy_from_slice(&one.Ainf);
        }
    }
}

/// The solved potentials of one ordinary frequency that the source formulation needs.
#[derive(Clone, Copy)]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub struct SourceInput<'a, R> {
    pub phi: &'a [Complex64], // [rhs][Nf]; LphiI on pressure surfaces
    pub u: &'a [Complex64],
    pub phiI: &'a [Complex64],
    pub dphiI: &'a [Complex64],
    pub LphiI: &'a [Complex64],
    pub Smat: &'a [Complex<R>],     // the character source matrices
    pub Vst: &'a [[Complex<R>; 3]], // grad of the source integrals at the sector centroids
    pub fpt: &'a [FinPoint],        // finite depth: the one-point factors per panel
    pub scs: Option<&'a Static<R>>, // the static rows at the control-surface points
}
