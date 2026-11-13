//! The field points of a problem (rule 7, reference/inputs-model-objects §3): the verdict on each
//! point (`PointVerdict`: in the fluid or a tank, above the free surface, below the sea bed, outside
//! the domain, on or inside a body, on a lid, near a low-order panel, ...) from the closed wetted
//! surface of each body and its solid angle, and the check that reports the first point that breaks
//! a rule.

use crate::codes::PointStatus;
use crate::constants::PI;
use crate::error::{Error, Location, Result, index};
use crate::geometry::{
    PLANE_TOL, Placement, Vec3, cross, dot, max_of, min_of, norm, reflect_vertices, reflections, v3,
};
use crate::mesh::{Mesh, patch_eval};
use crate::mesh_desc::PatchKind;
use crate::model_codes::Domain;
use crate::problem::{
    Problem, Role, all_records, body_label, finite3, placed_records, placement_of, plane_tol, records_of,
};

/// The verdict on a field point: its status, its body, lid or tank, and the message of
/// the rule it breaks; phase and sub give the order of the checks (the first one is reported).
#[derive(Clone, Debug)]
pub struct PointVerdict {
    pub status: PointStatus,
    pub index: i32,
    pub body: i32,
    pub lid: i32,
    pub phase: i32,
    pub sub: i32,
    pub message: String,
}

impl Default for PointVerdict {
    fn default() -> Self {
        PointVerdict {
            status: PointStatus::Fluid,
            index: -1,
            body: -1,
            lid: -1,
            phase: 0,
            sub: 0,
            message: String::new(),
        }
    }
}

// The closed wetted surface of a body as triangles: its wetted and pressure-surface elements
// with their reflections, placed, and their mirror images in z = 0 and in the walls of the domain,
// every triangle oriented as its element; patches subdivided (4 x 4 per span, at least 8 x 8).
struct Closure {
    tri: Vec<[Vec3; 3]>,
    lo: Vec3,
    hi: Vec3,
    panels: Vec<(Vec3, f64)>, // low order: centroid and diameter of every placed panel
}

#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn closure_of(m: &Mesh, role: &[Role], place: &Placement, domain: Domain) -> Closure {
    let mut c = Closure {
        tri: Vec::new(),
        lo: v3(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        hi: v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        panels: Vec::new(),
    };
    let mut base: Vec<[Vec3; 3]> = Vec::new();
    let mut add = |a: Vec3, mut b: Vec3, mut d: Vec3, flip: bool| {
        if flip {
            std::mem::swap(&mut b, &mut d);
        }
        base.push([a, b, d]);
    };
    for r in reflections(m.isx, m.isy) {
        for rec in 0..m.count as usize {
            let put = |x: Vec3| place.apply(v3(r.sx * x[0], r.sy * x[1], x[2]));
            if role[rec] == Role::Plate {
                // a free-surface plate closes nothing below z = 0 (it is its own image) but a
                // point just under it is near the body
                if !m.patch {
                    c.panels.push((put(m.shapes[rec].centre), m.shapes[rec].size));
                }
                continue;
            }
            if role[rec] != Role::Wet && role[rec] != Role::Pressure {
                continue;
            }
            let flip = r.sx * r.sy < 0.0;
            if !m.patch {
                let sh = &m.shapes[rec];
                let v: Vec<Vec3> = sh.hull.iter().map(|h| put(*h)).collect();
                let mut k = 1;
                while k + 1 < v.len() {
                    add(v[0], v[k], v[k + 1], flip);
                    k += 1;
                }
                if role[rec] == Role::Wet {
                    c.panels.push((put(sh.centre), sh.size));
                }
                continue;
            }
            let d = &m.discretization[rec];
            let nu = 8.max(4 * d.nu);
            let nv = 8.max(4 * d.nv);
            let mut g: Vec<Vec3> = Vec::new();
            for j in 0..=nv {
                for i in 0..=nu {
                    let u = -1.0 + 2.0 * i as f64 / nu as f64;
                    let w = -1.0 + 2.0 * j as f64 / nv as f64;
                    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                    if m.geometry == PatchKind::Flat && !m.trimmed() {
                        // the bilinear patch of its four vertices (patch_eval maps curved patches)
                        let q = &m.vertices[12 * rec..];
                        let n = [
                            0.25 * (1.0 - u) * (1.0 - w),
                            0.25 * (1.0 + u) * (1.0 - w),
                            0.25 * (1.0 + u) * (1.0 + w),
                            0.25 * (1.0 - u) * (1.0 + w),
                        ];
                        for k in 0..3 {
                            x[k] = n[0] * q[k] + n[1] * q[3 + k] + n[2] * q[6 + k] + n[3] * q[9 + k];
                        }
                    } else {
                        patch_eval(m, rec as i32, u, w, &mut x, &mut xu, &mut xv);
                    }
                    g.push(put(x));
                }
            }
            let (nu, nv) = (nu as usize, nv as usize);
            for j in 0..nv {
                for i in 0..nu {
                    let a = g[j * (nu + 1) + i];
                    let b = g[j * (nu + 1) + i + 1];
                    let e = g[(j + 1) * (nu + 1) + i + 1];
                    let f = g[(j + 1) * (nu + 1) + i];
                    add(a, b, e, flip);
                    add(a, e, f, flip);
                }
            }
        }
    }
    // the mirror images: z = 0 always (the free surface closes a surface-piercing body); the walls
    let mirrors = {
        let mut s: Vec<Vec3> = vec![v3(1.0, 1.0, 1.0), v3(1.0, 1.0, -1.0)];
        let twice = |s: &mut Vec<Vec3>, m2: Vec3| {
            let n = s.len();
            for k in 0..n {
                let e = v3(s[k][0] * m2[0], s[k][1] * m2[1], s[k][2] * m2[2]);
                s.push(e);
            }
        };
        if domain == Domain::XWall || domain == Domain::OrthogonalWalls {
            twice(&mut s, v3(-1.0, 1.0, 1.0));
        }
        if domain == Domain::YWall || domain == Domain::OrthogonalWalls {
            twice(&mut s, v3(1.0, -1.0, 1.0));
        }
        s
    };
    for mi in &mirrors {
        let flip = mi[0] * mi[1] * mi[2] < 0.0;
        for t in &base {
            let mut u = [Vec3::ZERO; 3];
            for k in 0..3 {
                u[k] = v3(mi[0] * t[k][0], mi[1] * t[k][1], mi[2] * t[k][2]);
            }
            if flip {
                u.swap(1, 2);
            }
            for x in &u {
                for k in 0..3 {
                    c.lo[k] = min_of(c.lo[k], x[k]);
                    c.hi[k] = max_of(c.hi[k], x[k]);
                }
            }
            c.tri.push(u);
        }
    }
    c
}

// The solid angle of a triangle seen from p (Van Oosterom and Strackee 1983); summed over a closed
// surface it is +-4 pi inside and 0 outside.
fn solid_angle(t: &[Vec3; 3], p: Vec3) -> f64 {
    let (a, b, c) = (t[0] - p, t[1] - p, t[2] - p);
    let (la, lb, lc) = (norm(a), norm(b), norm(c));
    let bc = cross(b, c);
    let num = dot(a, bc);
    let den = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
    2.0 * num.atan2(den)
}

// the state of the checks of classify_field_points: the order of the check and its body
struct Marker {
    phase: i32,
    sub: i32,
    body: i32,
}

fn mark(out: &mut [PointVerdict], c: &Marker, i: usize, status: PointStatus, index: i32, msg: String, lid: i32) {
    let p = &mut out[i];
    if !p.message.is_empty() {
        return; // an earlier check failed
    }
    p.status = status;
    p.index = index;
    p.body = c.body;
    p.lid = lid;
    p.phase = c.phase;
    p.sub = c.sub;
    p.message = msg;
}

impl Problem {
    pub fn exterior_point(&self, i: usize) -> bool {
        self.field_tanks.is_empty() || self.field_tanks[i] == 0
    }

    /// The verdict of the checks of check_field_points on every point (needs the rules of
    /// check_geometry_rules).
    // The checks of the field points (rule 7, reference/inputs-model-objects §3): the first rule each point breaks, in the order
    // of the checks. check_field_points reports the first of them all.
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    pub fn classify_field_points(&self, pts: &[Vec3], tanks: &[i32]) -> Vec<PointVerdict> {
        let mut out: Vec<PointVerdict> = vec![PointVerdict::default(); pts.len()];
        for i in 0..pts.len() {
            if !tanks.is_empty() && tanks[i] != 0 {
                out[i].status = PointStatus::Tank;
                out[i].index = tanks[i];
            }
        }
        let mut c = Marker { phase: 0, sub: 0, body: -1 };
        let exterior = |i: usize| tanks.is_empty() || tanks[i] == 0;
        for i in 0..pts.len() {
            if !finite3(&pts[i]) {
                mark(&mut out, &c, i, PointStatus::NotFinite, -1, format!("field point {} is not finite", i + 1), -1);
            }
        }
        // Field points: at z = 0 (free-surface elevation) or below it (pressure,
        // theory §4.1), above the bottom, in the fluid; points at z = 0 not on a body, a
        // lid or inside the water plane of a body (submerged points are not tested
        // against the bodies).
        let length = self.meshes[self.bodies[0].mesh as usize].reference_length;
        let ztol = PLANE_TOL * length;
        // points inside tanks: an existing tank with fluid, at or below its free surface
        // (body-local height plus the body's z); not tested against the tank walls, as submerged
        // exterior points are not tested against the bodies
        let mut ntank = 0;
        for b in &self.bodies {
            ntank += b.tanks.len() as i32;
        }
        c.phase = 1;
        for i in 0..tanks.len() {
            let k = tanks[i];
            if k == 0 {
                continue;
            }
            let at = format!("field point {}", i + 1);
            c.body = -1;
            if k < 0 || k > ntank {
                mark(
                    &mut out,
                    &c,
                    i,
                    PointStatus::NoTank,
                    k,
                    format!("{at}: tank {k} does not exist ({ntank} tanks)"),
                    -1,
                );
                continue;
            }
            let mut n = k;
            for bi in 0..self.bodies.len() {
                c.body = bi as i32;
                let b = &self.bodies[bi];
                if n > b.tanks.len() as i32 {
                    n -= b.tanks.len() as i32;
                    continue;
                }
                if b.tanks[(n - 1) as usize].density == 0.0 {
                    mark(
                        &mut out,
                        &c,
                        i,
                        PointStatus::EmptyTank,
                        k,
                        format!("{at} is in an empty tank (density_ratio 0)"),
                        -1,
                    );
                }
                let zt = self.derived_[bi].tank_z[(n - 1) as usize] + b.position[2];
                if pts[i][2] > zt + ztol {
                    mark(
                        &mut out,
                        &c,
                        i,
                        PointStatus::AboveTankSurface,
                        k,
                        format!("{at} lies above the free surface of its tank"),
                        -1,
                    );
                }
                break;
            }
        }
        c.body = -1;
        c.phase = 2;
        for i in 0..pts.len() {
            if !exterior(i) {
                continue;
            }
            let p = pts[i];
            let at = format!("field point {}", i + 1);
            if p[2] > ztol {
                mark(
                    &mut out,
                    &c,
                    i,
                    PointStatus::AboveFreeSurface,
                    -1,
                    format!("{at} lies above the free surface z = 0"),
                    -1,
                );
            }
            if self.env.depth.is_finite() && p[2] < -self.env.depth + ztol {
                mark(&mut out, &c, i, PointStatus::BelowSeaBed, -1, format!("{at} lies on or below the sea bed"), -1);
            }
            let fluid = match self.opt.domain {
                Domain::XWall => p[0] >= 0.0,
                Domain::YWall => p[1] >= 0.0,
                Domain::OrthogonalWalls => p[0] >= 0.0 && p[1] >= 0.0,
                Domain::Channel => p[1].abs() <= 0.5 * self.opt.channel_width,
                _ => true,
            };
            if !fluid {
                mark(&mut out, &c, i, PointStatus::OutsideDomain, -1, format!("{at} is outside the fluid domain"), -1);
            }
        }
        c.phase = 3;
        // waterline edges of the wetted elements (and sheets for the on-body test)
        struct Edge {
            a: Vec3,
            b: Vec3,
            closed: bool,
            tol: f64,
        }
        for bi in 0..self.bodies.len() {
            c.body = bi as i32;
            c.sub = c.body;
            let body = c.body;
            let b = &self.bodies[bi];
            let m = &self.meshes[b.mesh as usize];
            let place = placement_of(b);
            let mut edges: Vec<Edge> = Vec::new();
            // the elements of the body mesh with their roles, then the free-surface plates on
            // meshes of their own (the edges of a plate close it: the free surface is covered
            // there; its inner edges cancel in the winding)
            let mut groups: Vec<(&Mesh, Vec<Role>)> = vec![(m, self.derived_[bi].role.clone())];
            for pi in b.plates.iter().filter_map(|p| p.mesh) {
                let pm = &self.meshes[pi];
                groups.push((pm, vec![Role::Plate; pm.count as usize]));
            }
            for (m, roles) in &groups {
                let m: &Mesh = m;
                for r in reflections(m.isx, m.isy) {
                    for rec in 0..m.count as usize {
                        let role = roles[rec];
                        if role != Role::Wet && role != Role::Dipole && role != Role::Plate {
                            continue;
                        }
                        let s = &m.shapes[rec];
                        let tol = plane_tol(m, s);
                        let mut lp: Vec<Vec3> = Vec::new();
                        if !m.patch || m.geometry == PatchKind::Flat {
                            let v = [s.hull[0], s.hull[1], s.hull[2], s.hull[3]];
                            for q in reflect_vertices(&v, &r) {
                                lp.push(place.apply(q));
                            }
                        } else {
                            // boundary of the parameter square, sampled
                            let ns = 32;
                            let pt = |u: f64, v: f64| {
                                let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                                patch_eval(m, rec as i32, u, v, &mut x, &mut xu, &mut xv);
                                place.apply(v3(r.sx * x[0], r.sy * x[1], x[2]))
                            };
                            let mut sq: Vec<Vec3> = Vec::new();
                            let nsf = ns as f64;
                            for k in 0..ns {
                                sq.push(pt(-1.0 + 2.0 * k as f64 / nsf, -1.0));
                            }
                            for k in 0..ns {
                                sq.push(pt(1.0, -1.0 + 2.0 * k as f64 / nsf));
                            }
                            for k in 0..ns {
                                sq.push(pt(1.0 - 2.0 * k as f64 / nsf, 1.0));
                            }
                            for k in 0..ns {
                                sq.push(pt(-1.0, 1.0 - 2.0 * k as f64 / nsf));
                            }
                            if r.sx * r.sy < 0.0 {
                                sq.reverse();
                            }
                            lp = sq;
                        }
                        for k in 0..lp.len() {
                            let a = lp[k];
                            let cc = lp[(k + 1) % lp.len()];
                            if a[2].abs() <= tol && cc[2].abs() <= tol && norm(a - cc) > 0.0 {
                                // a point on an inner edge of a plate is not on the body: only its
                                // winding counts
                                if role == Role::Plate {
                                    edges.push(Edge { a, b: cc, closed: true, tol: -1.0 });
                                } else {
                                    edges.push(Edge { a, b: cc, closed: role == Role::Wet, tol });
                                }
                            }
                        }
                    }
                }
            }
            for i in 0..pts.len() {
                let p = pts[i];
                if p[2].abs() > ztol || !exterior(i) {
                    continue; // a submerged or tank point
                }
                let at = format!("field point {}", i + 1);
                let mut winding = 0.0;
                for e in &edges {
                    let (ax, ay, cx, cy) = (e.a[0] - p[0], e.a[1] - p[1], e.b[0] - p[0], e.b[1] - p[1]);
                    // distance from the point to the edge (horizontal)
                    let (ex, ey) = (cx - ax, cy - ay);
                    let mut t = -(ax * ex + ay * ey) / (ex * ex + ey * ey);
                    t = t.clamp(0.0, 1.0);
                    let (dx, dy) = (ax + t * ex, ay + t * ey);
                    if (dx * dx + dy * dy).sqrt() <= e.tol {
                        mark(&mut out, &c, i, PointStatus::OnBody, body, format!("{at} lies on {}", body_label(b)), -1);
                    }
                    if e.closed {
                        winding += (ax * cy - ay * cx).atan2(ax * cx + ay * cy);
                    }
                }
                // a tank problem's fluid is inside its waterline
                if winding.abs() > PI && !self.tank_domain {
                    mark(
                        &mut out,
                        &c,
                        i,
                        PointStatus::InsideWaterplane,
                        body,
                        format!("{at} lies inside the water plane of {}", body_label(b)),
                        -1,
                    );
                }
            }
            // submerged points inside the body, and low-order panels near a point
            if self.tank_domain {
                continue; // a tank problem's fluid is inside its body
            }
            let mut cl = closure_of(m, &self.derived_[bi].role, &place, self.opt.domain);
            for pi in b.plates.iter().filter_map(|p| p.mesh) {
                let pm = &self.meshes[pi];
                if pm.patch {
                    continue;
                }
                for r in reflections(pm.isx, pm.isy) {
                    for sh in &pm.shapes {
                        let x = sh.centre;
                        cl.panels.push((place.apply(v3(r.sx * x[0], r.sy * x[1], x[2])), sh.size));
                    }
                }
            }
            for i in 0..pts.len() {
                let p = pts[i];
                if p[2].abs() <= ztol || !exterior(i) || !out[i].message.is_empty() {
                    continue;
                }
                let mut in_box = true;
                for k in 0..3 {
                    in_box = in_box && p[k] >= cl.lo[k] && p[k] <= cl.hi[k];
                }
                if in_box {
                    let mut omega = 0.0;
                    for t in &cl.tri {
                        omega += solid_angle(t, p);
                    }
                    if omega.abs() > 2.0 * PI {
                        mark(
                            &mut out,
                            &c,
                            i,
                            PointStatus::InsideBody,
                            body,
                            format!("field point {} lies inside {}", i + 1, body_label(b)),
                            -1,
                        );
                        continue;
                    }
                }
                if out[i].status != PointStatus::Fluid {
                    continue;
                }
                for &(cc, d) in &cl.panels {
                    if norm(p - cc) < d {
                        out[i].status = PointStatus::NearBody;
                        out[i].index = body;
                        break;
                    }
                }
            }
        }
        c.body = -1;
        c.phase = 4;
        // points on a lid panel (horizontal containment in the convex panel)
        for (li, l) in self.lids.iter().enumerate() {
            c.sub = li as i32;
            let recs = if l.body >= 0 {
                let b = &self.bodies[l.body as usize];
                let m = &self.meshes[b.mesh as usize];
                placed_records(m, &records_of(&l.panels), &placement_of(b), &reflections(m.isx, m.isy))
            } else {
                let m = &self.meshes[l.mesh.expect("a lid of its own has its mesh (rule_lids)")];
                placed_records(m, &all_records(m), &Placement::default(), &reflections(m.isx, m.isy))
            };
            for i in 0..pts.len() {
                let p = pts[i];
                if p[2].abs() > ztol || !exterior(i) {
                    continue; // a submerged or tank point
                }
                for r in &recs {
                    let mut inn = true;
                    let n = r.hull.len();
                    let mut k = 0;
                    while k < n && inn {
                        let a = r.hull[k];
                        let cc = r.hull[(k + 1) % n];
                        k += 1;
                        let (ex, ey) = (cc[0] - a[0], cc[1] - a[1]);
                        let len = (ex * ex + ey * ey).sqrt();
                        if len == 0.0 {
                            continue;
                        }
                        if (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / len < -r.tol {
                            inn = false;
                        }
                    }
                    if inn {
                        mark(
                            &mut out,
                            &c,
                            i,
                            PointStatus::OnLid,
                            li as i32,
                            format!("field point {} lies on lids/{} (use the lid trace for lid points)", i + 1, l.name),
                            li as i32,
                        );
                    }
                }
            }
        }
        out
    }

    /// The field-point checks of rule 7 (reference/inputs-model-objects §3) on pts (global) with their tanks (empty: all
    /// exterior). Fails with ErrorKind::Value at the first failure, with the field point.
    pub fn check_field_points(&self, pts: &[Vec3], tanks: &[i32]) -> Result<()> {
        let v = self.classify_field_points(pts, tanks);
        let mut first = v.len();
        for i in 0..v.len() {
            if v[i].message.is_empty() || v[i].status == PointStatus::NotFinite {
                continue; // rule 6 checks finiteness
            }
            if first == v.len()
                || v[i].phase < v[first].phase
                || (v[i].phase == v[first].phase && v[i].sub < v[first].sub)
            {
                first = i;
            }
        }
        if first == v.len() {
            return Ok(());
        }
        let l = Location {
            field_point: Some(first),
            lid: index(v[first].lid),
            body: index(v[first].body),
            ..Location::default()
        };
        Err(Error::value(v[first].message.clone()).at(l))
    }
}
