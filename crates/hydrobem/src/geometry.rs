//! Low-order panel geometry (reference/inputs-meshes §1) and rigid placement: the projection onto the midpoint
//! plane, reflections, placement.

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use crate::constants::PI;

/// A point or vector of three doubles.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3(pub [f64; 3]);

pub const fn v3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3([x, y, z])
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3([0.0; 3]);
    pub fn as_ptr(&self) -> *const f64 {
        self.0.as_ptr()
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}
impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}
impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, b: Vec3) -> Vec3 {
        v3(self[0] + b[0], self[1] + b[1], self[2] + b[2])
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, b: Vec3) -> Vec3 {
        v3(self[0] - b[0], self[1] - b[1], self[2] - b[2])
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, a: Vec3) -> Vec3 {
        v3(self * a[0], self * a[1], self * a[2])
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        v3(-self[0], -self[1], -self[2])
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, b: Vec3) {
        *self = *self + b;
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, b: Vec3) {
        *self = *self - b;
    }
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    v3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
}
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// The larger of a and b, a on a tie or when either is NaN (unlike f64::max, which returns the
/// other argument of a NaN).
pub fn max_of(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// The smaller of a and b, a on a tie or when either is NaN (the counterpart of max_of).
pub fn min_of(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Relative tolerances of the panel rules.
pub const COINCIDENCE_TOL: f64 = 1e-6; // vertex coincidence / panel diameter
pub const CONVEXITY_TOL: f64 = 1e-6; // sine of the turning angle
pub const PLANE_TOL: f64 = 1e-6; // distance to a plane / reference length

/// The tolerance of a plane, waterline or coincidence test on a length scale L (m): PLANE_TOL L,
/// with a floor of 1 m (an absolute 1e-6 m for small panels and bodies).
pub fn length_tol(l: f64) -> f64 {
    PLANE_TOL * if l > 1.0 { l } else { 1.0 }
}

/// A low-order panel after the projection rule.
#[derive(Clone, Copy, Debug, Default)]
pub struct Panel {
    pub raw: [Vec3; 4],  // vertices as given
    pub proj: [Vec3; 4], // projected onto the midpoint plane
    pub distinct: i32,   // 3 or 4 distinct vertices
    pub poly: [Vec3; 4], // distinct projected vertices (first `distinct`)
    pub normal: Vec3,    // unit normal, right-hand rule of the vertex order
    pub area: f64,
    pub centroid: Vec3, // area centroid of the projected polygon
    pub diameter: f64,  // largest vertex distance
}

/// Build and validate one low-order panel (projection rule).
/// v12: four vertices, x y z each, raw order. Returns "" on success, else the reason (the caller
/// adds the element number).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn make_panel(v12: &[f64], out: &mut Panel) -> String {
    if v12[..12].iter().any(|v| !v.is_finite()) {
        return "non-finite vertex coordinate".into();
    }
    for i in 0..4 {
        out.raw[i] = v3(v12[3 * i], v12[3 * i + 1], v12[3 * i + 2]);
    }

    let mut diameter: f64 = 0.0;
    for i in 0..4 {
        for j in i + 1..4 {
            diameter = diameter.max(norm(out.raw[i] - out.raw[j]));
        }
    }
    out.diameter = diameter;
    if diameter == 0.0 {
        return "fewer than three distinct vertices".into();
    }
    let tol = COINCIDENCE_TOL * diameter;

    // Remove cyclically consecutive coincident vertices (a repeated vertex gives a triangle, reference/inputs-meshes §1).
    let mut keep = [0usize; 4];
    let mut m = 0;
    for i in 0..4 {
        let next = out.raw[(i + 1) % 4];
        if norm(out.raw[i] - next) > tol {
            keep[m] = i;
            m += 1;
        }
    }
    if m < 3 {
        return "fewer than three distinct vertices".into();
    }
    for a in 0..m {
        for b in a + 1..m {
            if norm(out.raw[keep[a]] - out.raw[keep[b]]) <= tol {
                return "projected polygon is self-intersecting (coincident non-adjacent vertices)".into();
            }
        }
    }

    // Plane through the midpoints of the four sides: its normal is the cross product of the
    // diagonals (Varignon parallelogram) and it contains the vertex mean.
    let c = 0.25 * (out.raw[0] + out.raw[1] + out.raw[2] + out.raw[3]);
    let n = cross(out.raw[2] - out.raw[0], out.raw[3] - out.raw[1]);
    let nn = norm(n);
    // Projected area is |d1 x d2| / 2 (the diagonals lie in the plane).
    if 0.5 * nn <= COINCIDENCE_TOL * diameter * diameter {
        return "zero area (degenerate or self-intersecting polygon)".into();
    }
    let nh = (1.0 / nn) * n;
    for i in 0..4 {
        out.proj[i] = out.raw[i] - dot(out.raw[i] - c, nh) * nh;
    }

    out.distinct = m as i32;
    for a in 0..m {
        out.poly[a] = out.proj[keep[a]];
    }
    for a in 0..m {
        let e1 = out.poly[(a + 1) % m] - out.poly[a];
        let e2 = out.poly[(a + 2) % m] - out.poly[(a + 1) % m];
        let s = dot(cross(e1, e2), nh) / (norm(e1) * norm(e2));
        if !(s > CONVEXITY_TOL) {
            return "projected polygon is not strictly convex or is self-intersecting".into();
        }
    }

    let mut area = 0.0;
    let mut moment = Vec3::ZERO;
    for a in 1..m - 1 {
        let t = 0.5 * dot(cross(out.poly[a] - out.poly[0], out.poly[a + 1] - out.poly[0]), nh);
        area += t;
        moment += (t / 3.0) * (out.poly[0] + out.poly[a] + out.poly[a + 1]);
    }
    out.area = area;
    out.centroid = (1.0 / area) * moment;
    out.normal = nh;
    String::new()
}

/// Reflection copies selected by the symmetry flags: index 0 is the stored panel, then
/// x-reflection, y-reflection, both (only those enabled).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reflection {
    pub sx: f64, // +1 or -1
    pub sy: f64,
}

pub fn reflections(isx: i32, isy: i32) -> Vec<Reflection> {
    let mut r = vec![Reflection { sx: 1.0, sy: 1.0 }];
    if isx != 0 {
        r.push(Reflection { sx: -1.0, sy: 1.0 });
    }
    if isy != 0 {
        r.push(Reflection { sx: 1.0, sy: -1.0 });
    }
    if isx != 0 && isy != 0 {
        r.push(Reflection { sx: -1.0, sy: -1.0 });
    }
    r
}

/// Reflects the raw vertices of a panel and reverses the vertex order when the reflection
/// changes orientation, so the normal keeps pointing into the fluid.
pub fn reflect_vertices(v: &[Vec3; 4], r: &Reflection) -> [Vec3; 4] {
    let mut w = [Vec3::ZERO; 4];
    for i in 0..4 {
        w[i] = v3(r.sx * v[i][0], r.sy * v[i][1], v[i][2]);
    }
    if r.sx * r.sy < 0.0 {
        return [w[0], w[3], w[2], w[1]];
    }
    w
}

/// Rigid placement x = Rz(yaw) x' + position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Placement {
    pub position: Vec3,
    pub yaw_deg: f64,
}

impl Placement {
    pub fn rotate(&self, p: Vec3) -> Vec3 {
        let a = self.yaw_deg * (PI / 180.0);
        let (c, s) = (a.cos(), a.sin());
        v3(c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])
    }
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.rotate(p) + self.position
    }
}
