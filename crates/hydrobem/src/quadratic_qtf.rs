//! The quadratic part of the second-order force and moment by pressure integration (theory §12.1):
//! the terms of the mean drift (pressure_drift.rs), each a time mean of a product of two
//! first-order quantities, taken for two waves (ω_a, β_a) and (ω_b, β_b), at their difference
//! and their sum frequency. Every frequency elements the first-order values the pressure drift
//! uses (QtfRecorder: the total potential and velocity at the points of the mean wetted surface
//! and its waterline, the motions); the pairs are formed once every frequency is solved. The
//! control-surface QTFs (theory §11.3) take the terms of the control-surface drift the same way,
//! with the two terms that are time derivatives (zero in the mean, Chen 2007 eq. 35, 36): over
//! the hull ρ ∂/∂t [∇Φ (X·n)] (the hull values of the pressure drift) and over the free surface
//! inside the control surface (ρ/g) ∂/∂t [Φ_t ∇Φ]. A body with generalized modes adds their terms
//! (pressure_drift.rs) and their rows, the generalized forces.

use num_complex::Complex64;

use crate::geometry::{Vec3, cross, v3};
use crate::model_codes::WaterlineTerm;
use crate::parallel::StepRows;
use crate::pressure_drift::{ClosedForms, ModeShapes, tangents};

const CZ: Complex64 = Complex64::new(0.0, 0.0);

/// A point of the mean wetted surface: about the body origin O and the normal into the fluid in
/// body axes, its global z and its quadrature weight.
#[derive(Clone, Debug)]
struct SurfacePoint {
    xb: Vec3,
    n: Vec3,
    z: f64,
    w: f64,
    s: Vec<Vec3>,          // the displacement of each generalized mode of the body there
    b: Vec<[[f64; 3]; 3]>, // and its gradient
}

/// A waterline element: its point about O, the normal of its panel (body axes), its length.
#[derive(Clone, Debug)]
struct WaterlinePoint {
    xb: Vec3,
    n: Vec3,
    dl: f64,
    s: Vec<Vec3>, // the displacement of each generalized mode of the body there
    plate: bool,  // under a free-surface plate: its values the potential and the plate edge's X_3
}

/// One integration of the pressure drift: a body's wetted surface, or the walls of one of its
/// tanks (with the tank's density and heights from its free surface), with the arguments of
/// PressureDrift::new and ::finish.
#[derive(Clone, Debug)]
struct Group {
    body: usize,
    tank: bool,
    rho: f64,
    closed: bool,
    rc: Vec3,
    zo: f64,
    gm: usize, // the generalized modes of the body
    surface: Vec<SurfacePoint>,
    waterline: Vec<WaterlinePoint>,
    hydrostatic: ClosedForms, // the sums of the closed forms of the hydrostatic part
}

impl Group {
    // the values per heading: the six motions and those of the generalized modes, then (phi, v)
    // per surface point and phi per waterline element (and the plate edge's X_3 under a plate)
    fn len(&self) -> usize {
        6 + self.gm + 4 * self.surface.len() + self.waterline.iter().map(|e| 1 + e.plate as usize).sum::<usize>()
    }
}

/// The first-order values of one frequency: the groups (their geometry, recorded with the first
/// heading) and the values of every heading, [heading][group values].
#[derive(Clone, Debug, Default)]
pub struct QtfSamples {
    omega: f64,
    groups: Vec<Group>,
    values: Vec<Vec<Complex64>>,
    cs_groups: Vec<CsGroup>, // the control surfaces of the bodies, with their values per heading
    cs_values: Vec<Vec<Complex64>>,
}

/// A point of a control surface: its part (0 submerged S_c, 1 free surface S_f, 2 the line CL),
/// about O, its normal away from the body (horizontal on CL) and its weight (area or length).
#[derive(Clone, Debug)]
struct CsPoint {
    part: u8,
    xb: Vec3,
    n: Vec3,
    w: f64,
}

/// The control surface of a body with the arguments of ControlSurfaceDrift::new, its body
/// waterline and the points of the hydrostatic closed forms (the wetted surface).
#[derive(Clone, Debug)]
struct CsGroup {
    body: usize,
    rho: f64,
    alternative: WaterlineTerm,
    rc: Vec3,
    zo: f64,
    points: Vec<CsPoint>,
    waterline: Vec<WaterlinePoint>,
    hydrostatic: ClosedForms, // the sums of the closed forms of the hydrostatic part
}

impl CsGroup {
    // the values per heading: the six motions, (phi, v) per point, phi per waterline element
    fn len(&self) -> usize {
        6 + 4 * self.points.len() + self.waterline.len()
    }
}

/// Elements the samples of one frequency while the pressure drift is computed: per heading, the
/// groups in the same order (the geometry with the first heading only).
#[derive(Debug, Default)]
pub struct QtfRecorder {
    s: QtfSamples,
    heading: usize,
    first: bool, // the first heading: the geometry is recorded
}

impl QtfRecorder {
    pub fn new(omega: f64, headings: usize) -> QtfRecorder {
        QtfRecorder {
            s: QtfSamples { omega, values: vec![Vec::new(); headings], ..QtfSamples::default() },
            heading: 0,
            first: true,
        }
    }
    /// The start of heading h (zero-based, in order).
    pub fn heading(&mut self, h: usize) {
        self.heading = h;
        self.first = h == 0;
    }
    /// A group of the heading: its body, whether it is a tank's walls, density, closed forms,
    /// rotation centre, the global z of O (zero for a tank, as its finish) and the six rigid
    /// motions of the body.
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    pub fn group(&mut self, body: usize, tank: bool, rho: f64, closed: bool, rc: &Vec3, zo: f64, x: &[Complex64; 6]) {
        if self.first {
            self.s.groups.push(Group {
                body,
                tank,
                rho,
                closed,
                rc: *rc,
                zo,
                gm: 0,
                surface: Vec::new(),
                waterline: Vec::new(),
                hydrostatic: ClosedForms::default(),
            });
        }
        self.s.values[self.heading].extend_from_slice(x);
    }
    /// The motions of the body's generalized modes, right after its group (none for a rigid body).
    pub fn generalized(&mut self, xg: &[Complex64]) {
        if self.first {
            self.current().gm = xg.len();
        }
        self.s.values[self.heading].extend_from_slice(xg);
    }
    fn current(&mut self) -> &mut Group {
        self.s.groups.last_mut().expect("a group begun")
    }
    /// A point of the mean wetted surface, as PressureDrift::surface.
    pub fn surface(&mut self, phi: Complex64, v: &[Complex64; 3], xb: &Vec3, n: &Vec3, z: f64, weight: f64) {
        self.surface_with(phi, v, xb, n, z, weight, None);
    }
    /// A point of the mean wetted surface, as PressureDrift::surface_with.
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
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
        if self.first {
            let (s, b) = shapes.map_or((Vec::new(), Vec::new()), |m| (m.s.to_vec(), m.b.to_vec()));
            self.current().surface.push(SurfacePoint { xb: *xb, n: *n, z, w: weight, s, b });
        }
        let vals = &mut self.s.values[self.heading];
        vals.push(phi);
        vals.extend_from_slice(v);
    }
    /// A waterline element, as PressureDrift::waterline.
    pub fn waterline(&mut self, phi: Complex64, xb: &Vec3, n: &Vec3, dl: f64) {
        self.waterline_with(phi, xb, n, dl, None);
    }
    /// A waterline element, as PressureDrift::waterline_with.
    pub fn waterline_with(&mut self, phi: Complex64, xb: &Vec3, n: &Vec3, dl: f64, shapes: Option<&[Vec3]>) {
        if self.first {
            let s = shapes.map_or(Vec::new(), |s| s.to_vec());
            self.current().waterline.push(WaterlinePoint { xb: *xb, n: *n, dl, s, plate: false });
        }
        self.s.values[self.heading].push(phi);
    }
    /// A waterline element under a free-surface plate, as PressureDrift::waterline_under_plate.
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    pub fn waterline_under_plate(
        &mut self,
        phi: Complex64,
        top: Complex64,
        xb: &Vec3,
        n: &Vec3,
        dl: f64,
        shapes: Option<&[Vec3]>,
    ) {
        if self.first {
            let s = shapes.map_or(Vec::new(), |s| s.to_vec());
            self.current().waterline.push(WaterlinePoint { xb: *xb, n: *n, dl, s, plate: true });
        }
        self.s.values[self.heading].extend_from_slice(&[phi, top]);
    }
    /// A point of the closed forms, as PressureDrift::hydrostatic_point (geometry only).
    pub fn hydrostatic_point(&mut self, x: &Vec3, n: &Vec3, z: f64, weight: f64, wetted: bool) {
        if self.first {
            self.current().hydrostatic.add(x, n, z, weight, wetted);
        }
    }
    /// The samples of the frequency.
    pub fn finish(self) -> QtfSamples {
        debug_assert!(self.s.values.iter().all(|v| v.len() == self.s.groups.iter().map(Group::len).sum::<usize>()));
        self.s
    }
}

/// Elements the control-surface values of one frequency while the control-surface drift is
/// computed (after the pressure drift), into the samples of the frequency.
#[derive(Debug, Default)]
pub struct CsRecorder {
    groups: Vec<CsGroup>,
    values: Vec<Vec<Complex64>>,
    heading: usize,
    first: bool,
}

impl CsRecorder {
    pub fn new(headings: usize) -> CsRecorder {
        CsRecorder { values: vec![Vec::new(); headings], first: true, ..CsRecorder::default() }
    }
    /// The start of heading h (zero-based, in order).
    pub fn heading(&mut self, h: usize) {
        self.heading = h;
        self.first = h == 0;
    }
    /// The control surface of a body: as ControlSurfaceDrift::new, with the global z of O.
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    pub fn group(&mut self, body: usize, rho: f64, alternative: WaterlineTerm, rc: &Vec3, zo: f64, x: &[Complex64; 6]) {
        if self.first {
            self.groups.push(CsGroup {
                body,
                rho,
                alternative,
                rc: *rc,
                zo,
                points: Vec::new(),
                waterline: Vec::new(),
                hydrostatic: ClosedForms::default(),
            });
        }
        self.values[self.heading].extend_from_slice(x);
    }
    fn current(&mut self) -> &mut CsGroup {
        self.groups.last_mut().expect("a group begun")
    }
    /// A point of part 0 (S_c), 1 (S_f) or 2 (CL): the total potential and velocity (body axes).
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    pub fn point(&mut self, part: i32, phi: Complex64, v: &[Complex64; 3], xb: &Vec3, n: &Vec3, weight: f64) {
        if self.first {
            self.current().points.push(CsPoint { part: part as u8, xb: *xb, n: *n, w: weight });
        }
        let vals = &mut self.values[self.heading];
        vals.push(phi);
        vals.extend_from_slice(v);
    }
    /// An element of the body waterline, as ControlSurfaceDrift::waterline.
    pub fn waterline(&mut self, phi: Complex64, xb: &Vec3, n: &Vec3, dl: f64) {
        if self.first {
            self.current().waterline.push(WaterlinePoint { xb: *xb, n: *n, dl, s: Vec::new(), plate: false });
        }
        self.values[self.heading].push(phi);
    }
    /// A wetted point of the hydrostatic closed forms (geometry only).
    pub fn hydrostatic_point(&mut self, x: &Vec3, n: &Vec3, z: f64, weight: f64) {
        if self.first {
            self.current().hydrostatic.add(x, n, z, weight, true);
        }
    }
    /// Into the samples of the frequency.
    pub fn finish(self, s: &mut QtfSamples) {
        debug_assert!(self.values.iter().all(|v| v.len() == self.groups.iter().map(CsGroup::len).sum::<usize>()));
        s.cs_groups = self.groups;
        s.cs_values = self.values;
    }
}

/// The product of two first-order quantities x(t) y(t) for the waves a and b: its component at
/// the difference frequency ω_a − ω_b, (x_a y_b* + x_b* y_a) / 4, or at the sum ω_a + ω_b,
/// (x_a y_b + x_b y_a) / 4 (the coefficient of A_a A_b* e^{i(ω_a − ω_b)t}, resp. A_a A_b
/// e^{i(ω_a + ω_b)t}, in the force Re Σ_a Σ_b over every ordered pair; for a = b the difference
/// is the mean <x y> = Re(x y*) / 2).
#[derive(Clone, Copy)]
struct Pair {
    sum: bool,
}

/// A quantity of the waves a and b.
type S = (Complex64, Complex64);

impl Pair {
    #[inline]
    fn p(self, x: S, y: S) -> Complex64 {
        if self.sum { 0.25 * (x.0 * y.1 + x.1 * y.0) } else { 0.25 * (x.0 * y.1.conj() + x.1.conj() * y.0) }
    }
}

fn add3(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [(a[0].0 + b[0].0, a[0].1 + b[0].1), (a[1].0 + b[1].0, a[1].1 + b[1].1), (a[2].0 + b[2].0, a[2].1 + b[2].1)]
}

// a x b of a complex and a real vector, for both waves
fn cross_sr(a: &[S; 3], b: &Vec3) -> [S; 3] {
    let c = |k: usize| -> S {
        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
        (a[i].0 * b[j] - a[j].0 * b[i], a[i].1 * b[j] - a[j].1 * b[i])
    };
    [c(0), c(1), c(2)]
}

/// The second-order rotation H of theory §11.2 (lower triangular) for the pair.
fn rotation2(p: Pair, al: &[S; 3]) -> [[Complex64; 3]; 3] {
    let aa = |i: usize, k: usize| p.p(al[i], al[k]);
    [
        [-0.5 * (aa(1, 1) + aa(2, 2)), CZ, CZ],
        [aa(0, 1), -0.5 * (aa(0, 0) + aa(2, 2)), CZ],
        [aa(0, 2), aa(1, 2), -0.5 * (aa(0, 0) + aa(1, 1))],
    ]
}

fn hv(h: &[[Complex64; 3]; 3], v: &Vec3) -> [Complex64; 3] {
    [h[0][0] * v[0], h[1][0] * v[0] + h[1][1] * v[1], h[2][0] * v[0] + h[2][1] * v[1] + h[2][2] * v[2]]
}

// the pair product of the cross product, <a × b> with each mean replaced by the pair product
fn pcross(p: Pair, a: &[S; 3], b: &[S; 3]) -> [Complex64; 3] {
    std::array::from_fn(|d| {
        let (i, j) = ((d + 1) % 3, (d + 2) % 3);
        p.p(a[i], b[j]) - p.p(a[j], b[i])
    })
}

// x × a of a real and a complex vector, for both waves
fn cross_rs(x: &Vec3, a: &[S; 3]) -> [S; 3] {
    let c = |k: usize| -> S {
        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
        (x[i] * a[j].0 - x[j] * a[i].0, x[i] * a[j].1 - x[j] * a[i].1)
    };
    [c(0), c(1), c(2)]
}

/// The force and moment about O (body axes) of one group for the waves a and b, then the
/// generalized forces of the body's generalized modes: the terms of PressureDrift (surface,
/// waterline, closed forms) with each mean <x y> replaced by the pair product. va, vb: the group's
/// values of the two waves; wa, wb their frequencies. The closed forms are PressureDrift's
/// (ClosedForms); the surface and waterline terms are written again here because the drift's mean
/// Re(a b*)/2 and the pair product at a = b round differently (a change to either is made to both;
/// the difference coefficient of a wave with itself is the drift, tested to 1e-12).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn group_pair(gr: &Group, va: &[Complex64], vb: &[Complex64], wa: f64, wb: f64, g: f64, sum: bool) -> Vec<Complex64> {
    let p = Pair { sum };
    let rho = gr.rho;
    let cf = gr.closed;
    let iw: S = (Complex64::new(0.0, wa), Complex64::new(0.0, wb));
    let al: [S; 3] = [(va[3], vb[3]), (va[4], vb[4]), (va[5], vb[5])];
    let aR = cross_sr(&al, &gr.rc);
    let xo: [S; 3] =
        [(va[0] - aR[0].0, vb[0] - aR[0].1), (va[1] - aR[1].0, vb[1] - aR[1].1), (va[2] - aR[2].0, vb[2] - aR[2].1)];
    let H = rotation2(p, &al);
    // H applied about R (PressureDrift): H (x − R), H R
    let at_o = gr.rc == Vec3::ZERO;
    let HR = if at_o { [CZ; 3] } else { hv(&H, &gr.rc) };
    let ng = gr.gm;
    let xgm: Vec<S> = (0..ng).map(|i| (va[6 + i], vb[6 + i])).collect(); // the generalized motions
    let mut F = vec![CZ; 6 + ng];
    let mut k = 6 + ng;
    for q in &gr.surface {
        let phi: S = (va[k], vb[k]);
        let v: [S; 3] = [(va[k + 1], vb[k + 1]), (va[k + 2], vb[k + 2]), (va[k + 3], vb[k + 3])];
        k += 4;
        let mut Dp = add3(xo, cross_sr(&al, &q.xb)); // displacement of the point
        // the generalized modes: Xg = Σ ξ_g S_g, Eg = Σ ξ_g B_g (pressure_drift.rs)
        let gmodes = (ng > 0 && !q.s.is_empty()).then(|| {
            let mut xg = [(CZ, CZ); 3];
            let mut eg = [[(CZ, CZ); 3]; 3];
            for (m, x) in xgm.iter().enumerate() {
                for i in 0..3 {
                    xg[i].0 += x.0 * q.s[m][i];
                    xg[i].1 += x.1 * q.s[m][i];
                    for l in 0..3 {
                        eg[i][l].0 += x.0 * q.b[m][i][l];
                        eg[i][l].1 += x.1 * q.b[m][i][l];
                    }
                }
            }
            (xg, eg)
        });
        if let Some((xg, _)) = &gmodes {
            Dp = add3(Dp, *xg);
        }
        let Hx = if at_o { hv(&H, &q.xb) } else { hv(&H, &(q.xb - gr.rc)) };
        let mut p2 = CZ;
        for d in 0..3 {
            let ivd: S = (iw.0 * v[d].0, iw.1 * v[d].1);
            p2 += 0.5 * p.p(v[d], v[d]) + p.p(Dp[d], ivd);
        }
        let p2 = -rho * (p2 + if cf { CZ } else { g * Hx[2] });
        let p1: S = {
            let a = |s: usize| {
                let (iws, ph, dz) = if s == 0 { (iw.0, phi.0, Dp[2].0) } else { (iw.1, phi.1, Dp[2].1) };
                -rho * (iws * ph + if cf { CZ } else { g * dz })
            };
            (a(0), a(1))
        };
        let p0 = if cf { 0.0 } else { -rho * g * q.z };
        let an = cross_sr(&al, &q.n);
        let xn = cross(q.xb, q.n);
        let axn = cross_sr(&al, &xn);
        let xin = cross_sr(&xo, &q.n);
        // <xi x (alpha x n)>: the pair products of the components
        let xa = |d: usize| {
            let (i, j) = ((d + 1) % 3, (d + 2) % 3);
            p.p(xo[i], an[j]) - p.p(xo[j], an[i])
        };
        let Hn = hv(&H, &q.n);
        let mut Hxn = hv(&H, &xn);
        if !at_o {
            // H (x × n) − (H R) × n
            for d in 0..3 {
                let (i, j) = ((d + 1) % 3, (d + 2) % 3);
                Hxn[d] -= HR[i] * q.n[j] - HR[j] * q.n[i];
            }
        }
        for d in 0..3 {
            let axin: S = (axn[d].0 + xin[d].0, axn[d].1 + xin[d].1);
            F[d] -= q.w * (p2 * q.n[d] + p.p(p1, an[d]) + p0 * Hn[d]);
            F[3 + d] -= q.w * (p2 * xn[d] + p.p(p1, axin) + p0 * (Hxn[d] + xa(d)));
        }
        let Some((xg, eg)) = gmodes else { continue };
        // the generalized parts of the normal and the moment, and the generalized forces, as
        // PressureDrift::surface_with
        let ev = |t: &Vec3| -> [S; 3] {
            std::array::from_fn(|i| {
                (
                    eg[i][0].0 * t[0] + eg[i][1].0 * t[1] + eg[i][2].0 * t[2],
                    eg[i][0].1 * t[0] + eg[i][1].1 * t[1] + eg[i][2].1 * t[2],
                )
            })
        };
        let tr: S = (eg[0][0].0 + eg[1][1].0 + eg[2][2].0, eg[0][0].1 + eg[1][1].1 + eg[2][2].1);
        let n1g: [S; 3] = std::array::from_fn(|c| {
            let et = |s: usize| {
                let e = |i: usize| if s == 0 { eg[i][c].0 } else { eg[i][c].1 };
                e(0) * q.n[0] + e(1) * q.n[1] + e(2) * q.n[2]
            };
            (tr.0 * q.n[c] - et(0), tr.1 * q.n[c] - et(1))
        });
        let (t1, t2) = tangents(&q.n);
        let (e1, e2) = (ev(&t1), ev(&t2));
        let (a1, a2) = (cross_sr(&al, &t1), cross_sr(&al, &t2));
        let (c1, c2, c3) = (pcross(p, &e1, &e2), pcross(p, &a1, &e2), pcross(p, &e1, &a2));
        let n2g: [Complex64; 3] = std::array::from_fn(|d| c1[d] + c2[d] + c3[d]);
        let xr = add3(xo, cross_sr(&al, &q.xb));
        let n1 = add3(an, n1g);
        let xn1g = cross_rs(&q.xb, &n1g);
        let xgn = cross_sr(&xg, &q.n);
        let (m1, m2) = (pcross(p, &xg, &n1), pcross(p, &xr, &n1g));
        for d in 0..3 {
            let xn2g = q.xb[(d + 1) % 3] * n2g[(d + 2) % 3] - q.xb[(d + 2) % 3] * n2g[(d + 1) % 3];
            let pm: S = (xn1g[d].0 + xgn[d].0, xn1g[d].1 + xgn[d].1);
            F[d] -= q.w * (p.p(p1, n1g[d]) + p0 * n2g[d]);
            F[3 + d] -= q.w * (p.p(p1, pm) + p0 * (xn2g + m1[d] + m2[d]));
        }
        let n2: [Complex64; 3] = std::array::from_fn(|d| Hn[d] + n2g[d]);
        for (m, s) in q.s.iter().enumerate() {
            let sn = s[0] * q.n[0] + s[1] * q.n[1] + s[2] * q.n[2];
            let sn1: S =
                (s[0] * n1[0].0 + s[1] * n1[1].0 + s[2] * n1[2].0, s[0] * n1[0].1 + s[1] * n1[1].1 + s[2] * n1[2].1);
            let sn2 = s[0] * n2[0] + s[1] * n2[1] + s[2] * n2[2];
            F[6 + m] -= q.w * (p2 * sn + p.p(p1, sn1) + p0 * sn2);
        }
    }
    for e in &gr.waterline {
        let phi: S = (va[k], vb[k]);
        k += 1;
        let zeta: S = (Complex64::new(0.0, -wa / g) * phi.0, Complex64::new(0.0, -wb / g) * phi.1);
        let ax = cross_sr(&al, &e.xb);
        let mut D: S = (xo[2].0 + ax[2].0, xo[2].1 + ax[2].1);
        if ng > 0 && !e.s.is_empty() {
            for (m, x) in xgm.iter().enumerate() {
                D.0 += x.0 * e.s[m][2];
                D.1 += x.1 * e.s[m][2];
            }
        }
        let sq = (1.0 - e.n[2] * e.n[2]).max(1e-12).sqrt();
        if e.plate {
            // the band under a free-surface plate (PressureDrift::waterline_under_plate): the
            // pressure head less X_3 and zeta_r = top - X_3
            debug_assert!(!cf);
            let top: S = (va[k], vb[k]);
            k += 1;
            let a: S = (zeta.0 - D.0, zeta.1 - D.1);
            let zr: S = (top.0 - D.0, top.1 - D.1);
            let sw = -rho * g * (p.p(a, zr) - 0.5 * p.p(zr, zr)) * e.dl / sq;
            let xn = cross(e.xb, e.n);
            for d in 0..3 {
                F[d] += sw * e.n[d];
                F[3 + d] += sw * xn[d];
            }
            for (m, s) in e.s.iter().enumerate().take(ng) {
                F[6 + m] += sw * (s[0] * e.n[0] + s[1] * e.n[1] + s[2] * e.n[2]);
            }
            continue;
        }
        let zr: S = (zeta.0 - D.0, zeta.1 - D.1);
        let sw = -0.5 * rho * g * p.p(zr, zr) * e.dl / sq;
        let swh = if cf { -0.5 * rho * g * (p.p(zeta, zeta) - 2.0 * p.p(zeta, D)) * e.dl / sq } else { sw };
        let nh = v3(e.n[0], e.n[1], 0.0);
        let nv = v3(0.0, 0.0, e.n[2]);
        let xh = cross(e.xb, nh);
        let xv = cross(e.xb, nv);
        for d in 0..3 {
            F[d] += swh * nh[d] + sw * nv[d];
            F[3 + d] += swh * xh[d] + sw * xv[d];
        }
        for (m, s) in e.s.iter().enumerate().take(ng) {
            F[6 + m] += sw * (s[0] * e.n[0] + s[1] * e.n[1] + s[2] * e.n[2]);
        }
    }
    if cf {
        // the closed forms of the hydrostatic part (PressureDrift::finish) with the pair products
        let q = |i: usize| if i < 3 { xo[i] } else { al[i - 3] };
        let rg = rho * g;
        let [mut f2, mut f3, mut f4] = gr.hydrostatic.terms(gr.zo, |i, k| p.p(q(i), q(k)));
        if !at_o {
            let [t2, t3, t4] = gr.hydrostatic.translation([-HR[0], -HR[1], -HR[2]]);
            f2 += t2;
            f3 += t3;
            f4 += t4;
        }
        F[2] -= rg * f2;
        F[3] += rg * f3;
        F[4] += rg * f4;
    }
    F
}

/// The control-surface force and moment about O of one body for the waves a and b: the terms of
/// ControlSurfaceDrift with each mean replaced by the pair product, the hydrostatic closed forms,
/// and the two time derivatives (iΩ, Ω = ω_a ∓ ω_b) over the free surface inside the control
/// surface and over the hull (`hull`: the body's pressure-drift group with its values).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn cs_pair(
    cg: &CsGroup,
    va: &[Complex64],
    vb: &[Complex64],
    hull: Option<(&Group, &[Complex64], &[Complex64])>,
    wa: f64,
    wb: f64,
    g: f64,
    sum: bool,
) -> [Complex64; 6] {
    let p = Pair { sum };
    let rho = cg.rho;
    let iO = Complex64::new(0.0, if sum { wa + wb } else { wa - wb }); // d/dt of the pair
    let iw: S = (Complex64::new(0.0, wa), Complex64::new(0.0, wb));
    let al: [S; 3] = [(va[3], vb[3]), (va[4], vb[4]), (va[5], vb[5])];
    let aR = cross_sr(&al, &cg.rc);
    let xo: [S; 3] =
        [(va[0] - aR[0].0, vb[0] - aR[0].1), (va[1] - aR[1].0, vb[1] - aR[1].1), (va[2] - aR[2].0, vb[2] - aR[2].1)];
    let mut F = [CZ; 6];
    let addf = |F: &mut [Complex64; 6], f: [Complex64; 3], x: &Vec3| {
        for d in 0..3 {
            F[d] += f[d];
        }
        F[3] += x[1] * f[2] - x[2] * f[1];
        F[4] += x[2] * f[0] - x[0] * f[2];
        F[5] += x[0] * f[1] - x[1] * f[0];
    };
    let mut k = 6;
    for q in &cg.points {
        let phi: S = (va[k], vb[k]);
        let v: [S; 3] = [(va[k + 1], vb[k + 1]), (va[k + 2], vb[k + 2]), (va[k + 3], vb[k + 3])];
        k += 4;
        let zeta: S = (-iw.0 / g * phi.0, -iw.1 / g * phi.1);
        let vv = p.p(v[0], v[0]) + p.p(v[1], v[1]) + p.p(v[2], v[2]);
        match q.part {
            0 => {
                // -rho int [grad phi dphi/dn - n |grad phi|^2 / 2]
                let vn: S = (
                    v[0].0 * q.n[0] + v[1].0 * q.n[1] + v[2].0 * q.n[2],
                    v[0].1 * q.n[0] + v[1].1 * q.n[1] + v[2].1 * q.n[2],
                );
                let f = [0, 1, 2].map(|d| -rho * q.w * (p.p(v[d], vn) - 0.5 * q.n[d] * vv));
                addf(&mut F, f, &q.xb);
            }
            1 => {
                // rho k int [zeta dphi_t/dz + |grad phi|^2 / 2]
                let ivz: S = (iw.0 * v[2].0, iw.1 * v[2].1);
                let fz = rho * q.w * (p.p(zeta, ivz) + 0.5 * vv);
                addf(&mut F, [CZ, CZ, fz], &q.xb);
                if cg.alternative == WaterlineTerm::Body {
                    let gx: S = (-iw.0 / g * v[0].0, -iw.1 / g * v[0].1);
                    let gy: S = (-iw.0 / g * v[1].0, -iw.1 / g * v[1].1);
                    let f = [-rho * g * q.w * p.p(zeta, gx), -rho * g * q.w * p.p(zeta, gy), CZ];
                    addf(&mut F, f, &q.xb);
                }
                // the time derivative (rho / g) d/dt [phi_t grad phi]
                let pt: S = (iw.0 * phi.0, iw.1 * phi.1);
                let f = [0, 1, 2].map(|d| rho / g * q.w * iO * p.p(pt, v[d]));
                addf(&mut F, f, &q.xb);
            }
            _ => {
                if cg.alternative != WaterlineTerm::Body {
                    let s = -0.5 * rho * g * p.p(zeta, zeta) * q.w;
                    addf(&mut F, [s * q.n[0], s * q.n[1], CZ], &q.xb);
                }
            }
        }
    }
    for e in &cg.waterline {
        let phi: S = (va[k], vb[k]);
        k += 1;
        let zeta: S = (-iw.0 / g * phi.0, -iw.1 / g * phi.1);
        let ax = cross_sr(&al, &e.xb);
        let D = add3(xo, ax);
        let h = (e.n[0] * e.n[0] + e.n[1] * e.n[1]).max(1e-24).sqrt();
        let (nhx, nhy, nz) = (-e.n[0] / h, -e.n[1] / h, -e.n[2]);
        let Dn: S = (D[0].0 * nhx + D[1].0 * nhy, D[0].1 * nhx + D[1].1 * nhy);
        let sq = (1.0 - nz * nz).max(1e-12).sqrt();
        let zr: S = (zeta.0 - D[2].0, zeta.1 - D[2].1);
        let fz = (-rho * g * p.p(zeta, Dn) + 0.5 * rho * g * nz / sq * p.p(zr, zr)) * e.dl;
        addf(&mut F, [CZ, CZ, fz], &e.xb);
        if cg.alternative == WaterlineTerm::Body {
            let s = 0.5 * rho * g * p.p(zeta, zeta) * e.dl;
            addf(&mut F, [s * nhx, s * nhy, CZ], &e.xb);
        }
    }
    // the hydrostatic part: the closed forms of the pressure drift
    let hs = Group {
        body: cg.body,
        tank: false,
        rho,
        closed: true,
        rc: cg.rc,
        zo: cg.zo,
        gm: 0,
        surface: Vec::new(),
        waterline: Vec::new(),
        hydrostatic: cg.hydrostatic,
    };
    let fs = group_pair(&hs, &va[..6], &vb[..6], wa, wb, g, sum);
    for d in 0..6 {
        F[d] += fs[d];
    }
    // the hull: rho d/dt [grad phi (X . n)], n into the fluid
    if let Some((hg, ha, hb)) = hull {
        let mut k = 6 + hg.gm;
        for q in &hg.surface {
            let v: [S; 3] = [(ha[k + 1], hb[k + 1]), (ha[k + 2], hb[k + 2]), (ha[k + 3], hb[k + 3])];
            k += 4;
            let Dp = add3(xo, cross_sr(&al, &q.xb));
            let Dn: S = (
                Dp[0].0 * q.n[0] + Dp[1].0 * q.n[1] + Dp[2].0 * q.n[2],
                Dp[0].1 * q.n[0] + Dp[1].1 * q.n[1] + Dp[2].1 * q.n[2],
            );
            let f = [0, 1, 2].map(|d| rho * q.w * iO * p.p(v[d], Dn));
            addf(&mut F, f, &q.xb);
        }
    }
    F
}

/// The coefficients of every pair of frequencies (a <= b; with `diagonal` a = b only) and of
/// headings of each body, both orders filled: f(a, b, ha, hb, body) gives the difference and the sum
/// coefficients of the body's widths[body] columns (2 widths[body] values), laid out body after
/// body. The rows a on the frequency workers of the solve (`rows`: each row an item of its
/// second-order step, its pairs in parallel on the worker's inner threads; a row not computed after
/// a cancel stays NaN, `StepRows::finish` returns the cancel).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn all_pairs(
    rows: &StepRows,
    F: usize,
    H: usize,
    widths: &[usize],
    pairs: bool,
    diagonal: bool,
    f: &(dyn Fn(usize, usize, usize, usize, usize) -> Vec<Complex64> + Sync),
) -> (Vec<Complex64>, Vec<Complex64>) {
    let NB = widths.len();
    let n6: usize = widths.iter().sum();
    let first: Vec<usize> = widths.iter().scan(0, |s, &w| Some(std::mem::replace(s, *s + w))).collect();
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let mut dif = vec![nan; F * F * H * H * n6];
    let mut sum = dif.clone();
    // the rows a, each with the columns b >= a (independent of the thread count)
    let keys = |a: usize| {
        let mut row: Vec<(usize, usize, usize, usize)> = Vec::new();
        for b in a..if diagonal { a + 1 } else { F } {
            for ha in 0..H {
                // the pair (a, ha; a, hb) of hb < ha is the conjugate of (a, hb; a, ha)
                for hb in (if b == a { ha } else { 0 })..H {
                    if !pairs && ha != hb {
                        continue;
                    }
                    for bi in 0..NB {
                        row.push((b, ha, hb, bi));
                    }
                }
            }
        }
        row
    };
    let part = |a: usize, &(b, ha, hb, bi): &(usize, usize, usize, usize)| -> PairValues {
        (b, ha * H + hb, bi, f(a, b, ha, hb, bi))
    };
    let rows = rows.rows(F, &keys, &part);
    for (a, row) in rows.into_iter().enumerate() {
        for (b, hh, bi, o) in row {
            let (ha, hb) = (hh / H, hh % H);
            let w = widths[bi];
            for c in 0..w {
                let at =
                    |x: usize, y: usize, hx: usize, hy: usize| (((x * F + y) * H + hx) * H + hy) * n6 + first[bi] + c;
                dif[at(b, a, hb, ha)] = o[c].conj();
                sum[at(b, a, hb, ha)] = o[w + c];
                dif[at(a, b, ha, hb)] = o[c]; // last: on the diagonal the computed value
                sum[at(a, b, ha, hb)] = o[w + c];
            }
        }
    }
    (dif, sum)
}

// the values of group number gi of the samples of one heading (the offsets of the groups before it)
fn offset(lens: impl Iterator<Item = usize>, gi: usize) -> usize {
    lens.take(gi).sum()
}

/// The control-surface QTFs of every body, as quadratic_qtfs; the hull term from the
/// pressure-drift values of the same frequencies.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the samples, their sizes and the options of the pairs
pub fn control_surface_qtfs(
    rows: &StepRows,
    samples: &[QtfSamples],
    H: usize,
    NB: usize,
    gravity: f64,
    pairs: bool,
    diagonal: bool,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let f = |a: usize, b: usize, ha: usize, hb: usize, bi: usize| -> Vec<Complex64> {
        let (sa, sb) = (&samples[a], &samples[b]);
        let mut o = vec![CZ; 12];
        for (ci, cg) in sa.cs_groups.iter().enumerate() {
            if cg.body != bi {
                continue;
            }
            let k = offset(sa.cs_groups.iter().map(CsGroup::len), ci);
            let n = cg.len();
            let va = &sa.cs_values[ha][k..k + n];
            let vb = &sb.cs_values[hb][k..k + n];
            // the body's wetted surface among the pressure-drift groups (not its tanks)
            let hull = sa.groups.iter().position(|g| g.body == bi && !g.tank).map(|gi| {
                let k = offset(sa.groups.iter().map(Group::len), gi);
                let n = sa.groups[gi].len();
                (&sa.groups[gi], &sa.values[ha][k..k + n], &sb.values[hb][k..k + n])
            });
            let d = cs_pair(cg, va, vb, hull, sa.omega, sb.omega, gravity, false);
            let s = cs_pair(cg, va, vb, hull, sa.omega, sb.omega, gravity, true);
            for c in 0..6 {
                o[c] += d[c];
                o[6 + c] += s[c];
            }
        }
        o
    };
    all_pairs(rows, samples.len(), H, &vec![6; NB], pairs, diagonal, &f)
}

// a computed pair: the column b, the heading pair ha * H + hb, the body, the difference and the sum
// coefficients of its columns
type PairValues = (usize, usize, usize, Vec<Complex64>);

/// The quadratic QTFs of every body for the frequencies of `samples` (one per ordinary frequency,
/// in order) and H headings: the difference and the sum coefficients [F][F][H][H][6 NB] (body
/// axes, the moment about O; the results turn it to the rotation centre), both pair orders filled
/// (Q⁻(b, a) = Q⁻(a, b)*, Q⁺(b, a) = Q⁺(a, b)); with `pairs` false only the equal headings (the others are NaN); with `diagonal` only the equal
/// frequencies (the others are NaN). `generalized`: the generalized modes of each body; their
/// generalized forces are the last two arrays, [F][F][H][H][G] (G their sum, the bodies in order;
/// empty without generalized modes). The rows of the pairs on the frequency workers (`rows`, all_pairs).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::type_complexity)] // the difference and sum arrays of the rigid and generalized rows
#[allow(clippy::too_many_arguments)] // the samples, their sizes and the options of the pairs
pub fn quadratic_qtfs(
    rows: &StepRows,
    samples: &[QtfSamples],
    H: usize,
    generalized: &[usize],
    gravity: f64,
    pairs: bool,
    diagonal: bool,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let widths: Vec<usize> = generalized.iter().map(|g| 6 + g).collect();
    let f = |a: usize, b: usize, ha: usize, hb: usize, bi: usize| -> Vec<Complex64> {
        let (sa, sb) = (&samples[a], &samples[b]);
        let w = widths[bi];
        let mut o = vec![CZ; 2 * w];
        let (mut ka, mut kb) = (0, 0);
        for (ga, gb) in sa.groups.iter().zip(&sb.groups) {
            let n = ga.len();
            debug_assert_eq!(n, gb.len());
            if ga.body == bi {
                let va = &sa.values[ha][ka..ka + n];
                let vb = &sb.values[hb][kb..kb + n];
                let d = group_pair(ga, va, vb, sa.omega, sb.omega, gravity, false);
                let s = group_pair(ga, va, vb, sa.omega, sb.omega, gravity, true);
                for c in 0..d.len().min(w) {
                    o[c] += d[c];
                    o[w + c] += s[c];
                }
            }
            ka += n;
            kb += n;
        }
        o
    };
    let (dif, sum) = all_pairs(rows, samples.len(), H, &widths, pairs, diagonal, &f);
    let G: usize = generalized.iter().sum();
    if G == 0 {
        return (dif, sum, Vec::new(), Vec::new());
    }
    // the rigid rows and the generalized rows apart
    let nw: usize = widths.iter().sum();
    let split = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
        let (mut r, mut g) =
            (Vec::with_capacity(v.len() / nw * 6 * widths.len()), Vec::with_capacity(v.len() / nw * G));
        for row in v.chunks(nw) {
            let mut k = 0;
            for &w in &widths {
                r.extend_from_slice(&row[k..k + 6]);
                g.extend_from_slice(&row[k + 6..k + w]);
                k += w;
            }
        }
        (r, g)
    };
    let (d6, dg) = split(&dif);
    let (s6, sg) = split(&sum);
    (d6, s6, dg, sg)
}
