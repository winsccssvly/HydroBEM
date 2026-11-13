//! The low-order geometry of a solve: the frame and its panels with their reflections and wall
//! images, the sectors and characters of the symmetry group, the surfaces of the panels (body,
//! lids, interior surfaces) and the control-surface points.
//!
//! An element group holds the source of each of its panels (`Src`: an element of a mesh, a panel of
//! an interior surface or of the absorbed lid panels), resolved by the builder.

use crate::error::{Error, Result};
use crate::geometry::{
    COINCIDENCE_TOL, Panel, Placement, Reflection, Vec3, cross, length_tol, make_panel, norm, reflect_vertices,
    reflections, v3,
};
use crate::ho_geometry::patch_geo;
use crate::interior::{InteriorPanels, interior_panels};
use crate::low_order::{CsPoint, DamperSurf, Geometry, PressSurf, SurfaceKind};
use crate::mesh::Mesh;
use crate::model_codes::{Domain, RestoringConvention};
use crate::panel_integrals::make_panel_geom;
use crate::problem::{Body, Problem, Role};
use crate::quadrature::gauss_rule;
use crate::route::wall_wavemaker;

/// Characters of a reflection group (e, sigma_x, sigma_y, sigma_x sigma_y order).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn characters(r: &[Reflection]) -> Vec<Vec<f64>> {
    let G = r.len();
    if G == 1 {
        return vec![vec![1.0]];
    }
    if G == 2 {
        return vec![vec![1.0, 1.0], vec![1.0, -1.0]];
    }
    // e, sigma_x, sigma_y, sigma_x sigma_y; characters (s1, s2)
    vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, -1.0, 1.0, -1.0], vec![1.0, 1.0, -1.0, -1.0], vec![1.0, -1.0, -1.0, 1.0]]
}

/// Local reflection of body b induced by the global reflection g, if the body
/// is symmetric under g: the body lies on the global
/// symmetry plane and one of its own symmetry planes coincides with it.
pub fn induced_reflection(b: &Body, isx: i32, isy: i32, g: &Reflection, local: &mut Reflection) -> bool {
    let mut yaw = b.yaw % 360.0;
    if yaw < 0.0 {
        yaw += 360.0;
    }
    let is = |a: f64| (yaw - a).abs() <= 1e-12 || (yaw - a - 360.0).abs() <= 1e-12;
    let straight = is(0.0) || is(180.0);
    let turned = is(90.0) || is(270.0);
    *local = Reflection { sx: 1.0, sy: 1.0 };
    for axis in 0..2 {
        let flip = if axis == 0 { g.sx < 0.0 } else { g.sy < 0.0 };
        if !flip {
            continue;
        }
        if b.position[axis] != 0.0 || !(straight || turned) {
            return false;
        }
        // a global x reflection is the local x reflection when straight, the local y one when turned
        let use_local_x = (axis == 0) == straight;
        if use_local_x {
            if isx == 0 {
                return false;
            }
            local.sx = -local.sx;
        } else {
            if isy == 0 {
                return false;
            }
            local.sy = -local.sy;
        }
    }
    true
}

/// The source of a panel of an element group: element r of mesh i, panel k
/// of the interior surface of body b, or panel k of the absorbed external lid panels.
#[derive(Clone, Copy, Debug)]
enum Src {
    Mesh(usize, usize),
    Interior(usize, usize),
    Store(usize),
}

// Panels of one owner (a body, or an external lid mesh in global
// coordinates) expanded by one set of local reflections: the wetted elements
// and the body's lid elements in element order, then the interior panels when
// they share the symmetry form.
#[derive(Clone, Debug, Default)]
struct RecordGroup {
    panels: Vec<Src>,
    kind: Vec<SurfaceKind>,
    lid: Vec<i32>, // lid index and position in the lid's element list
    lidpos: Vec<i32>,
    lidout: Vec<[i32; 4]>, // explicit output index per reflection (-1: from lidpos)
    pss: Vec<i32>,         // pressure surface (index into Geometry::ps), else -1
    dmp: Vec<i32>,         // damper (index into Geometry::dmp), else -1
    isx: i32,
    isy: i32,
}

impl RecordGroup {
    fn add(&mut self, p: Src, k: SurfaceKind, l: i32, pos: i32, out: [i32; 4], ps: i32) {
        self.panels.push(p);
        self.kind.push(k);
        self.lid.push(l);
        self.lidpos.push(pos);
        self.lidout.push(out);
        self.pss.push(ps);
        self.dmp.push(-1);
    }
    // add with no lid, position, outward neighbours or pressure surface
    fn add1(&mut self, p: Src, k: SurfaceKind) {
        self.add(p, k, -1, 0, [-1, -1, -1, -1], -1);
    }
}

// An external lid of a single body in open sea, given in global coordinates
// with its own symmetry flags, is taken into the body frame when its panel set
// is invariant under the body's mesh reflections there and no panel is mapped
// onto itself: one representative per orbit is added to
// the body's elements, with the output index of every image, so that the solve
// keeps the body's symmetry sectors. Returns false (nothing changed) otherwise.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn absorb_external_lid(
    m: &Problem,
    l: i32,
    b: &Body,
    rg: &mut RecordGroup,
    lid_off: &[i32],
    nrec: i32,
    store: &mut Vec<Panel>,
) -> bool {
    let lm = &m.meshes[m.lids[l as usize].mesh.expect("a lid of its own has its mesh (rule_lids)")];
    let inv = Placement { position: Vec3::ZERO, yaw_deg: -b.yaw };
    struct Item {
        v: [Vec3; 4], // body-frame vertices
        c: Vec3,
        size: f64,
        out: i32,
    }
    let mut items: Vec<Item> = Vec::new();
    let own = reflections(lm.isx, lm.isy);
    for ri in 0..own.len() {
        for rec in 0..lm.count {
            let p = &lm.panels[rec as usize];
            let mut v = reflect_vertices(&p.raw, &own[ri]);
            for q in v.iter_mut() {
                *q = inv.rotate(v3(q[0] - b.position[0], q[1] - b.position[1], 0.0));
            }
            let c = inv.rotate(v3(
                own[ri].sx * p.centroid[0] - b.position[0],
                own[ri].sy * p.centroid[1] - b.position[1],
                0.0,
            ));
            items.push(Item { v, c, size: p.diameter, out: lid_off[l as usize] + ri as i32 * nrec + rec });
        }
    }
    let grp = reflections(rg.isx, rg.isy);
    if grp.len() == 1 {
        return false; // nothing to gain
    }
    let find = |c: Vec3, tol: f64| -> i32 {
        let mut hit = -1;
        for (i, it) in items.iter().enumerate() {
            if (it.c[0] - c[0]).abs() <= tol && (it.c[1] - c[1]).abs() <= tol {
                if hit >= 0 {
                    return -2; // ambiguous
                }
                hit = i as i32;
            }
        }
        hit
    };
    let mut orbit_of = vec![-1i32; items.len()];
    let mut orbits: Vec<[i32; 4]> = Vec::new();
    for i in 0..items.len() {
        if orbit_of[i] >= 0 {
            continue;
        }
        let mut members = [-1i32; 4];
        let tol = COINCIDENCE_TOL * items[i].size;
        for gi in 0..grp.len() {
            let j = find(v3(grp[gi].sx * items[i].c[0], grp[gi].sy * items[i].c[1], 0.0), tol);
            if j < 0 || orbit_of[j as usize] >= 0 {
                return false;
            }
            if members[..gi].contains(&j) {
                return false; // a panel mapped onto itself
            }
            members[gi] = j;
        }
        for gi in 0..grp.len() {
            orbit_of[members[gi] as usize] = orbits.len() as i32;
        }
        orbits.push(members);
    }
    for o in &orbits {
        let mut p = Panel::default();
        let mut v12 = [0.0f64; 12];
        for k in 0..4 {
            for d in 0..3 {
                v12[3 * k + d] = items[o[0] as usize].v[k][d];
            }
        }
        if !make_panel(&v12, &mut p).is_empty() {
            return false;
        }
        store.push(p);
        let mut out = [-1i32; 4];
        for gi in 0..grp.len() {
            out[gi] = items[o[gi] as usize].out;
        }
        rg.add(Src::Store(store.len() - 1), SurfaceKind::Lid, l, 0, out, -1);
    }
    true
}

#[derive(Clone, Debug)]
struct Owner {
    place: Body, // placement (an external lid: the global frame)
    bi: i32,
    groups: Vec<RecordGroup>,
}

/// The control-surface points of every body in the frame of (yaw, px, py):
/// panel centroids or patch Gauss points of S_c and S_f, and the elements of CL, expanded by
/// the symmetry flags of each control surface (also used by the higher order).
pub fn control_surface_points(m: &Problem, yaw: f64, px: f64, py: f64) -> Vec<CsPoint> {
    let mut cs: Vec<CsPoint> = Vec::new();
    let inv = Placement { position: Vec3::ZERO, yaw_deg: -yaw };
    // the control surfaces, expanded by their symmetry flags: the panel centroids of the
    // submerged part S_c and of the part in z = 0 (S_f), and the edges of S_c in z = 0 (CL)
    for bi in 0..m.bodies.len() as i32 {
        let b = &m.bodies[bi as usize];
        let cm = &m.meshes[b.control_surface.expect("a control surface on every body (the rules)")];
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        let frame = |xl: Vec3| {
            let xg = place.apply(xl);
            inv.rotate(v3(xg[0] - px, xg[1] - py, xg[2]))
        };
        let tol = length_tol(cm.reference_length);
        let zb = b.position[2]; // the free surface is z = -zb in body axes
        let at = CsAt { body: bi, zb, tol, frame: &frame };
        if cm.patch {
            for pt in 0..cm.count {
                patch_control_points(cm, pt, &at, &mut cs);
            }
        } else {
            panel_control_points(cm, &at, &mut cs);
        }
    }
    cs
}

// The body of a control surface, the height of the free surface in its axes (z = -zb), the length
// tolerance of its mesh and the map of its body axes to the frame.
struct CsAt<'a> {
    body: i32,
    zb: f64,
    tol: f64,
    frame: &'a dyn Fn(Vec3) -> Vec3,
}

// A higher-order control surface (theory §11.3): patch pt divided into nu x nv panels, iquo x iquo
// Gauss points on each; the sides of the submerged patches in z = 0 are CL, with iquo Gauss points
// per panel side.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn patch_control_points(cm: &Mesh, pt: i32, at: &CsAt, cs: &mut Vec<CsPoint>) {
    let (bi, zb, tol, frame) = (at.body, at.zb, at.tol, at.frame);
    let pg = patch_geo(cm, pt);
    let d = &cm.discretization[pt as usize];
    let (gx, gw) = gauss_rule(d.iquo.max(1));
    // the point, the unit normal into the fluid, the area element and the tangents at (u, v)
    let eval = |u: f64, v: f64| {
        let z = Vec3::ZERO;
        let (mut x, mut xu, mut xv) = (z, z, z);
        pg.eval(u, v, &mut x, &mut xu, &mut xv);
        let mut n = pg.nsign() * cross(xu, xv);
        let J = norm(n);
        if J > 0.0 {
            n = (1.0 / J) * n;
        }
        (x, n, J, xu, xv)
    };
    // whether the points t = -1, -1/2, ..., 1 of a line of the patch lie in z = 0
    let in_plane = |at_t: &dyn Fn(f64) -> (f64, f64)| {
        (0..=4).all(|k| {
            let (u, v) = at_t(-1.0 + 0.5 * k as f64);
            (eval(u, v).0[2] + zb).abs() <= tol
        })
    };
    // the whole patch in z = 0: S_f
    let flat = (0..=4).all(|iu| in_plane(&|t| (-1.0 + 0.5 * iu as f64, t)));
    for r in reflections(cm.isx, cm.isy) {
        let refl = |x: Vec3| v3(r.sx * x[0], r.sy * x[1], x[2]);
        for iv in 0..d.nv {
            for iu in 0..d.nu {
                for ib in 0..gx.len() {
                    for a in 0..gx.len() {
                        let u = -1.0 + (2.0 * iu as f64 + 1.0 + gx[a]) / d.nu as f64;
                        let v = -1.0 + (2.0 * iv as f64 + 1.0 + gx[ib]) / d.nv as f64;
                        let (x, n, J, _, _) = eval(u, v);
                        if !(J > 0.0) {
                            continue;
                        }
                        let wq = J * gw[a] * gw[ib] / (d.nu as f64 * d.nv as f64);
                        let c = refl(x);
                        let part = if flat { 1 } else { 0 };
                        cs.push(CsPoint { x: frame(c), xb: c, nb: refl(n), w: wq, part, body: bi });
                    }
                }
            }
        }
        if flat {
            continue;
        }
        for side in 0..4 {
            // 0: v = -1, 1: v = +1 (u varies); 2: u = -1, 3: u = +1 (v varies)
            let along_u = side < 2;
            let fixed = if side % 2 == 0 { -1.0 } else { 1.0 };
            let uv = |t: f64| if along_u { (t, fixed) } else { (fixed, t) };
            if !in_plane(&uv) {
                continue;
            }
            let np = if along_u { d.nu } else { d.nv };
            for k in 0..np {
                for a in 0..gx.len() {
                    let (u, v) = uv(-1.0 + (2.0 * k as f64 + 1.0 + gx[a]) / np as f64);
                    // a degenerate point (a collapsed patch side): just inside it
                    let (x, n, J, xu, xv) = eval(u, v);
                    let (x, n, xu, xv) = if J > 0.0 {
                        (x, n, xu, xv)
                    } else {
                        let (x, n, _, xu, xv) = eval(u * (1.0 - 1e-7), v * (1.0 - 1e-7));
                        (x, n, xu, xv)
                    };
                    let h = (n[0] * n[0] + n[1] * n[1]).sqrt();
                    if !(h > 0.0) {
                        continue;
                    }
                    let dl = norm(if along_u { xu } else { xv }) * gw[a] / np as f64;
                    let c = refl(v3(x[0], x[1], -zb));
                    let nb = refl(v3(n[0] / h, n[1] / h, 0.0));
                    cs.push(CsPoint { x: frame(c), xb: c, nb, w: dl, part: 2, body: bi });
                }
            }
        }
    }
}

// A low-order control surface: the panel centroids (S_c, or S_f for a panel in z = 0) and the
// panel edges of S_c in z = 0 (CL), every reflection of the symmetry flags.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn panel_control_points(cm: &Mesh, at: &CsAt, cs: &mut Vec<CsPoint>) {
    let (bi, zb, tol, frame) = (at.body, at.zb, at.tol, at.frame);
    for r in reflections(cm.isx, cm.isy) {
        for p in &cm.panels {
            let c = v3(r.sx * p.centroid[0], r.sy * p.centroid[1], p.centroid[2]);
            // the normal out of the fluid between the body and the control surface (away from
            // the body): the momentum balance of that fluid
            let n = v3(r.sx * p.normal[0], r.sy * p.normal[1], p.normal[2]);
            let flat = p.raw.iter().all(|v| (v[2] + zb).abs() <= tol);
            cs.push(CsPoint { x: frame(c), xb: c, nb: n, w: p.area, part: if flat { 1 } else { 0 }, body: bi });
            if flat {
                continue;
            }
            for e in 0..4 {
                let a = p.raw[e];
                let q = p.raw[(e + 1) % 4];
                if (a[2] + zb).abs() > tol || (q[2] + zb).abs() > tol || norm(q - a) <= tol {
                    continue;
                }
                let mid = v3(0.5 * r.sx * (a[0] + q[0]), 0.5 * r.sy * (a[1] + q[1]), -zb);
                let h = (n[0] * n[0] + n[1] * n[1]).sqrt();
                if !(h > 0.0) {
                    continue; // a horizontal panel has no waterline direction
                }
                let nb = v3(n[0] / h, n[1] / h, 0.0);
                cs.push(CsPoint { x: frame(mid), xb: mid, nb, w: norm(q - a), part: 2, body: bi });
            }
        }
    }
}

// The geometry of a low-order problem, built in stages (build_geometry below calls them in order).
struct Builder<'a> {
    m: &'a Problem,
    ints: &'a mut Vec<InteriorPanels>,
    g: Geometry,
    nb: i32,
    wx: bool,
    wy: bool,
    owners: Vec<Owner>,
    sym_ok: bool,
    external_lid: bool,
    external_sheet: bool, // a fixed damper as an owner of its own
    // lid output layout: lid by lid, reflections in reflections() order, elements in order
    lid_off: Vec<i32>,
    lid_rec: Vec<i32>,
    dmp_first: Vec<i32>,  // the first damper and pressure surface of each body
    fixed_dmp_first: i32, // the first of the fixed dampers in no body's frame
    ps_first: Vec<i32>,
    lid_store: Vec<Panel>, // external lid panels absorbed into a body's frame
    proj: Vec<[Vec3; 4]>,  // global projected vertices of every panel
}

impl<'a> Builder<'a> {
    fn new(m: &'a Problem, ints: &'a mut Vec<InteriorPanels>) -> Builder<'a> {
        let nb = m.bodies.len() as i32;
        let d = m.opt.domain;
        *ints = vec![InteriorPanels::default(); nb as usize];
        let owners = vec![Owner { place: Body::default(), bi: -1, groups: Vec::new() }; nb as usize];
        Builder {
            m,
            ints,
            g: Geometry::default(),
            nb,
            wx: d == Domain::XWall || d == Domain::OrthogonalWalls,
            wy: d == Domain::YWall || d == Domain::OrthogonalWalls,
            owners,
            sym_ok: true,
            external_lid: false,
            external_sheet: false,
            lid_off: vec![0; m.lids.len() + 1],
            lid_rec: vec![0; m.lids.len()],
            dmp_first: Vec::new(),
            fixed_dmp_first: 0,
            ps_first: Vec::new(),
            lid_store: Vec::new(),
            proj: Vec::new(),
        }
    }

    fn panel(&self, s: Src) -> &Panel {
        match s {
            Src::Mesh(i, r) => &self.m.meshes[i].panels[r],
            Src::Interior(b, k) => &self.ints[b].panels[k],
            Src::Store(k) => &self.lid_store[k],
        }
    }

    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn lid_layout(&mut self) {
        let counts = self.m.lid_panel_counts();
        for l in 0..self.m.lids.len() {
            self.lid_off[l + 1] = self.lid_off[l] + counts[l];
        }
        self.g.L = *self.lid_off.last().unwrap();
    }

    // The dipole dampers and the pressure surfaces: the pressure coordinates after the
    // body's six rigid and its generalized ones.
    fn dampers_and_pressure_surfaces(&mut self) -> Result<()> {
        let m = self.m;
        let nb = self.nb as usize;
        self.dmp_first = vec![0; nb];
        for bi in 0..nb {
            self.dmp_first[bi] = self.g.dmp.len() as i32;
            for d in &m.bodies[bi].dampers {
                self.g.dmp.push(DamperSurf { eps: d.epsilon, free: d.free(), porous: d.porous() });
            }
        }
        // the fixed dampers in no body's frame: at rest, impervious, in no force
        self.fixed_dmp_first = self.g.dmp.len() as i32;
        for d in &m.fixed_dampers {
            self.g.dmp.push(DamperSurf { eps: d.epsilon, free: true, porous: false });
        }
        // pressure surfaces
        self.ps_first = vec![0; nb];
        for bi in 0..nb {
            let b = &m.bodies[bi];
            self.ps_first[bi] = self.g.ps.len() as i32;
            let der = &m.derived()[bi];
            let mut coord = der.coordinate_offset + 6 + der.generalized;
            for (s, p) in b.pressure.iter().enumerate() {
                let mut q = PressSurf {
                    body: bi as i32,
                    coord0: coord,
                    id: s as i32 + 1,
                    chamber: p.chamber_coupling,
                    numeric: p.convention == RestoringConvention::NumericCompatibility,
                    ..Default::default()
                };
                for pm in &p.modes {
                    q.shapes.push(pm.shape);
                }
                let pmesh = &m.meshes[p.mesh.unwrap_or(b.mesh as usize)];
                // a pressure surface in z = 0: the centroids of all its panels in the plane (the
                // plane test of theory §14, as in the higher order); one with some panels in it
                // and some not is refused
                let tol = length_tol(pmesh.reference_length);
                let (mut inside, mut outside) = (false, false);
                for r in 0..pmesh.count as usize {
                    if p.mesh.is_some() || (der.role[r] == Role::Pressure && der.owner[r] == s as i32) {
                        let z0 = (pmesh.panels[r].centroid[2] + b.position[2]).abs() <= tol;
                        inside |= z0;
                        outside |= !z0;
                    }
                }
                if inside && outside {
                    return Err(Error::value(format!(
                        "body '{}': pressure surface {}: some of its panels lie in z = 0 and some do not (a \
                         pressure surface lies in the free surface or below it)",
                        b.name,
                        s + 1
                    )));
                }
                q.z0 = inside;
                self.g.ps.push(q);
                coord += p.modes.len() as i32;
            }
        }
        Ok(())
    }

    // The element groups of every body: wetted, sheet, pressure, lid and interior panels.
    fn body_owners(&mut self) -> Result<()> {
        let m = self.m;
        for bi in 0..self.nb as usize {
            let b = &m.bodies[bi];
            let mi = b.mesh as usize;
            let mesh = &m.meshes[mi];
            let der = &m.derived()[bi];
            self.owners[bi].place = b.clone();
            self.owners[bi].bi = bi as i32;
            let mut wet = RecordGroup { isx: mesh.isx, isy: mesh.isy, ..Default::default() };
            for r in 0..mesh.count as usize {
                let role = der.role[r];
                if role == Role::Wet {
                    wet.add1(Src::Mesh(mi, r), SurfaceKind::Wet);
                }
                if role == Role::Dipole && der.tank[r] < 0 {
                    // sheets in a tank: its own problem
                    wet.add1(Src::Mesh(mi, r), SurfaceKind::Sheet);
                    let d = der.owner[r];
                    if d >= 0 {
                        *wet.dmp.last_mut().unwrap() = self.dmp_first[bi] + d;
                    }
                }
                if role == Role::Pressure {
                    wet.add(
                        Src::Mesh(mi, r),
                        SurfaceKind::Press,
                        -1,
                        0,
                        [-1, -1, -1, -1],
                        self.ps_first[bi] + der.owner[r],
                    );
                }
                if role == Role::Plate {
                    wet.add1(Src::Mesh(mi, r), SurfaceKind::Plate);
                }
                if role == Role::Lid {
                    let l = der.owner[r];
                    let pos = self.lid_rec[l as usize];
                    self.lid_rec[l as usize] += 1;
                    wet.add(Src::Mesh(mi, r), SurfaceKind::Lid, l, pos, [-1, -1, -1, -1], -1);
                }
            }
            if let Some(di) = b.dipole_mesh {
                for r in 0..m.meshes[di].count as usize {
                    wet.add1(Src::Mesh(di, r), SurfaceKind::Sheet);
                }
            }
            // (a sheet in a symmetry plane of its mesh is rejected by select_route)
            let mut extra_groups: Vec<RecordGroup> = Vec::new();
            for (s, ps) in b.pressure.iter().enumerate() {
                let Some(pi) = ps.mesh else {
                    continue;
                };
                let pm = &m.meshes[pi];
                let same = pm.isx == mesh.isx && pm.isy == mesh.isy;
                if !same {
                    self.sym_ok = false; // an external pressure mesh with its own symmetry form: no sectors
                    extra_groups.push(RecordGroup { isx: pm.isx, isy: pm.isy, ..Default::default() });
                }
                let rg = if same { &mut wet } else { extra_groups.last_mut().unwrap() };
                for r in 0..pm.count as usize {
                    rg.add(Src::Mesh(pi, r), SurfaceKind::Press, -1, 0, [-1, -1, -1, -1], self.ps_first[bi] + s as i32);
                }
            }
            // the free-surface plates on meshes of their own, as the pressure surfaces
            for p in &b.plates {
                let Some(pi) = p.mesh else {
                    continue;
                };
                let pm = &m.meshes[pi];
                let same = pm.isx == mesh.isx && pm.isy == mesh.isy;
                if !same {
                    self.sym_ok = false; // a plate mesh with its own symmetry form: no sectors
                    extra_groups.push(RecordGroup { isx: pm.isx, isy: pm.isy, ..Default::default() });
                }
                let rg = if same { &mut wet } else { extra_groups.last_mut().unwrap() };
                for r in 0..pm.count as usize {
                    rg.add1(Src::Mesh(pi, r), SurfaceKind::Plate);
                }
            }
            self.ints[bi] = interior_panels(m, bi as i32)?;
            let ip = &self.ints[bi];
            let o = &mut self.owners[bi];
            if ip.panels.is_empty() || (ip.isx == mesh.isx && ip.isy == mesh.isy) {
                for k in 0..ip.panels.len() {
                    wet.add1(Src::Interior(bi, k), SurfaceKind::Interior);
                }
                o.groups.push(wet);
                o.groups.extend(extra_groups);
            } else {
                self.sym_ok = false; // the interior mesh has its own symmetry form: no sectors
                o.groups.push(wet);
                let mut inn = RecordGroup { isx: ip.isx, isy: ip.isy, ..Default::default() };
                for k in 0..ip.panels.len() {
                    inn.add1(Src::Interior(bi, k), SurfaceKind::Interior);
                }
                o.groups.push(inn);
                o.groups.extend(extra_groups);
            }
        }
        Ok(())
    }

    // The external lids as owners of their own.
    fn external_lids(&mut self) {
        let m = self.m;
        for l in 0..m.lids.len() {
            if m.lids[l].body >= 0 {
                continue;
            }
            self.external_lid = true;
            let li = m.lids[l].mesh.expect("a lid of its own has its mesh (rule_lids)");
            let lm = &m.meshes[li];
            let place = Body { position: Vec3::ZERO, yaw: 0.0, ..Body::default() };
            let mut rg = RecordGroup { isx: lm.isx, isy: lm.isy, ..Default::default() };
            for r in 0..lm.count {
                rg.add(Src::Mesh(li, r as usize), SurfaceKind::Lid, l as i32, r, [-1, -1, -1, -1], -1);
            }
            self.lid_rec[l] = lm.count;
            self.owners.push(Owner { place, bi: -1, groups: vec![rg] });
        }
    }

    // The fixed dampers in no body's frame: sheets of owners of their own, at rest in global
    // coordinates.
    fn fixed_dampers(&mut self) {
        let m = self.m;
        for (k, d) in m.fixed_dampers.iter().enumerate() {
            self.external_sheet = true;
            let mi = d.mesh as usize;
            let dm = &m.meshes[mi];
            let place = Body { position: Vec3::ZERO, yaw: 0.0, ..Body::default() };
            let mut rg = RecordGroup { isx: dm.isx, isy: dm.isy, ..Default::default() };
            for r in 0..dm.count as usize {
                rg.add1(Src::Mesh(mi, r), SurfaceKind::Sheet);
                *rg.dmp.last_mut().unwrap() = self.fixed_dmp_first + k as i32;
            }
            self.owners.push(Owner { place, bi: -1, groups: vec![rg] });
        }
    }

    fn absorb_lids(&mut self) {
        let m = self.m;
        // a single body in open sea absorbs symmetric external lids into its frame
        if self.nb == 1
            && !self.wx
            && !self.wy
            && self.external_lid
            && !self.external_sheet
            && self.owners[0].groups.len() == 1
        {
            let mut trial = self.owners[0].groups[0].clone();
            let mut all = true;
            let mut l = 0;
            while l < m.lids.len() && all {
                if m.lids[l].body < 0 {
                    all = absorb_external_lid(
                        m,
                        l as i32,
                        &m.bodies[0],
                        &mut trial,
                        &self.lid_off,
                        self.lid_rec[l],
                        &mut self.lid_store,
                    );
                }
                l += 1;
            }
            if all {
                self.owners[0].groups[0] = trial;
                self.owners.truncate(1);
                self.external_lid = false;
            }
        }
    }

    // One panel of an element group in the frame: its geometry, kind, outputs and rigid normals.
    fn add_panel(&mut self, rg: &RecordGroup, k: usize, ri: i32, r: &Reflection, place: &Placement, o: &Owner) {
        let m = self.m;
        let src = rg.panels[k];
        let p = *self.panel(src);
        let kind = rg.kind[k];
        let mut v = reflect_vertices(&p.proj, r);
        for q in v.iter_mut() {
            *q = place.apply(*q);
        }
        let mut nl = v3(r.sx * p.normal[0], r.sy * p.normal[1], p.normal[2]);
        let cl = v3(r.sx * p.centroid[0], r.sy * p.centroid[1], p.centroid[2]);
        let mut c = place.apply(cl);
        if kind == SurfaceKind::Interior || kind == SurfaceKind::Lid {
            // interior surfaces and lids lie exactly in z = 0;
            // lid elements oriented downward are taken counter-clockwise from above
            if p.normal[2] < 0.0 {
                v.swap(1, 3);
            }
            for q in v.iter_mut() {
                q[2] = 0.0;
            }
            c[2] = 0.0;
            nl = v3(0.0, 0.0, 1.0);
        }
        if kind == SurfaceKind::Plate {
            // a free-surface plate lies exactly in z = 0, its normal down into the water (elements
            // given upwards are taken clockwise from above)
            if p.normal[2] > 0.0 {
                v.swap(1, 3);
            }
            for q in v.iter_mut() {
                q[2] = 0.0;
            }
            c[2] = 0.0;
            nl = v3(0.0, 0.0, -1.0);
        }
        let psi = rg.pss[k];
        if kind == SurfaceKind::Press && self.g.ps[psi as usize].z0 {
            // a pressure surface in z = 0 is placed exactly there, normal downwards
            for q in v.iter_mut() {
                q[2] = 0.0;
            }
            c[2] = 0.0;
            nl = v3(0.0, 0.0, -1.0);
        }
        let g = &mut self.g;
        g.pg.push(make_panel_geom(&v, &place.rotate(nl), &c, p.area));
        let mut rv = reflect_vertices(&p.raw, r);
        for q in rv.iter_mut() {
            *q = place.apply(*q);
        }
        g.raw.push(rv);
        self.proj.push(v);
        g.body.push(o.bi);
        g.kind.push(kind);
        g.mirror.push(0);
        {
            // the element of the body mesh (outputs on the body panels); the elements of the
            // pressure surfaces and free-surface plates on meshes of their own are numbered on
            // after those of the body mesh, the pressure surfaces' meshes first, then the plates'
            // (reference/results)
            let mut rec = -1;
            if o.bi >= 0 {
                if let Src::Mesh(mi, ri_) = src {
                    let b = &m.bodies[o.bi as usize];
                    if mi == b.mesh as usize {
                        rec = ri_ as i32;
                    } else if kind == SurfaceKind::Press || kind == SurfaceKind::Plate {
                        let own =
                            b.pressure.iter().filter_map(|p| p.mesh).chain(b.plates.iter().filter_map(|p| p.mesh));
                        let mut next = m.meshes[b.mesh as usize].count;
                        for om in own {
                            if om == mi {
                                rec = next + ri_ as i32;
                                break;
                            }
                            next += m.meshes[om].count;
                        }
                    }
                }
            }
            g.rec.push(rec);
            g.lref.push(*r);
        }
        g.lid.push(rg.lid[k]);
        g.psurf.push(psi);
        g.damper.push(rg.dmp[k]);
        g.lc.push(cl);
        let po = if kind == SurfaceKind::Press {
            g.Q += 1;
            g.Q - 1
        } else {
            -1
        };
        g.press_out.push(po);
        g.lid_out.push(if kind != SurfaceKind::Lid {
            -1
        } else if rg.lidout[k][0] >= 0 {
            rg.lidout[k][ri as usize]
        } else {
            self.lid_off[rg.lid[k] as usize] + ri * self.lid_rec[rg.lid[k] as usize] + rg.lidpos[k]
        });
        if kind == SurfaceKind::Sheet {
            g.sheets = true;
        }
        if kind == SurfaceKind::Plate {
            g.plates = true;
        }
        if kind != SurfaceKind::Wet && kind != SurfaceKind::Sheet && kind != SurfaceKind::Plate {
            // the rigid modes do not move a pressure surface (theory §9.3)
            g.nj.push([0.0; 6]);
        } else {
            let rr = cl - o.place.rotation_center;
            let mom = cross(rr, nl);
            g.nj.push([nl[0], nl[1], nl[2], mom[0], mom[1], mom[2]]);
        }
    }

    // The frame, the group and the panels; g.Nf.
    fn place_panels(&mut self) {
        let m = self.m;
        self.g.W = if m.opt.domain == Domain::Channel { m.opt.channel_width } else { 0.0 };
        self.g.Rc = m.opt.channel_reflect;
        let owners = std::mem::take(&mut self.owners);
        if self.nb == 1
            && !self.wx
            && !self.wy
            && !self.external_lid
            && !self.external_sheet
            && owners[0].groups.len() == 1
            && self.g.W == 0.0
        {
            self.body_frame(&owners[0]);
        } else {
            self.global_frame(&owners);
        }
        self.owners = owners;
        self.g.Nf = self.g.pg.len() as i32;
    }

    // A single body: its horizontal frame; the mesh reflections form the group.
    fn body_frame(&mut self, o: &Owner) {
        let b = &o.place;
        let rg = &o.groups[0];
        self.g.yaw = b.yaw;
        self.g.px = b.position[0];
        self.g.py = b.position[1];
        self.g.refl = reflections(rg.isx, rg.isy);
        self.g.G = self.g.refl.len() as i32;
        let frame = Placement { position: v3(0.0, 0.0, b.position[2]), yaw_deg: 0.0 };
        self.g.Ns = rg.panels.len() as i32;
        for ri in 0..self.g.G {
            let r = self.g.refl[ri as usize];
            for k in 0..rg.panels.len() {
                self.add_panel(rg, k, ri, &r, &frame, o);
            }
        }
        for k in 0..self.g.Ns {
            self.g.sec.push(k);
        }
        for gi in 0..self.g.G {
            for k in 0..self.g.Ns {
                self.g.img.push(gi * self.g.Ns + k);
            }
        }
        self.g.walls = vec![Reflection { sx: 1.0, sy: 1.0 }];
    }

    // The global frame: the global reflections mapping every owner onto itself (G0) with the
    // sector panels and their images, the wall images, and the group of both.
    fn global_frame(&mut self, owners: &[Owner]) {
        let (wx, wy) = (self.wx, self.wy);
        let mut gx = self.sym_ok && !wx;
        let mut gy = self.sym_ok && !wy;
        for o in owners {
            let mut l = Reflection { sx: 1.0, sy: 1.0 };
            let rg = &o.groups[0];
            gx = gx && induced_reflection(&o.place, rg.isx, rg.isy, &Reflection { sx: -1.0, sy: 1.0 }, &mut l);
            gy = gy && induced_reflection(&o.place, rg.isx, rg.isy, &Reflection { sx: 1.0, sy: -1.0 }, &mut l);
        }
        let g0 = reflections(gx as i32, gy as i32);
        let img0 = self.owner_sectors(owners, &g0);
        self.g.Ns = self.g.sec.len() as i32;
        let nbase = self.wall_images();
        let g = &mut self.g;
        g.refl = reflections((gx || wx) as i32, (gy || wy) as i32);
        g.G = g.refl.len() as i32;
        for e in g.refl.clone() {
            let wpart = Reflection { sx: if wx { e.sx } else { 1.0 }, sy: if wy { e.sy } else { 1.0 } };
            let spart = Reflection { sx: if wx { 1.0 } else { e.sx }, sy: if wy { 1.0 } else { e.sy } };
            let (mut wi, mut si) = (0usize, 0usize);
            while g.walls[wi].sx != wpart.sx || g.walls[wi].sy != wpart.sy {
                wi += 1;
            }
            while g0[si].sx != spart.sx || g0[si].sy != spart.sy {
                si += 1;
            }
            for &k in &img0[si][..g.Ns as usize] {
                g.img.push(wi as i32 * nbase as i32 + k);
            }
        }
    }

    // The panels of every owner placed in the global frame, its sector panels (g.sec: one local
    // reflection per orbit of G0) and their images under every element of G0 ([g][sector panel]).
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn owner_sectors(&mut self, owners: &[Owner], g0: &[Reflection]) -> Vec<Vec<i32>> {
        let mut img0: Vec<Vec<i32>> = vec![Vec::new(); g0.len()];
        for o in owners {
            let place = Placement { position: o.place.position, yaw_deg: o.place.yaw };
            for rg in &o.groups {
                let local = reflections(rg.isx, rg.isy);
                let first = self.g.pg.len() as i32;
                for ri in 0..local.len() {
                    for k in 0..rg.panels.len() {
                        self.add_panel(rg, k, ri as i32, &local[ri], &place, o);
                    }
                }
                let index_of = |sx: f64, sy: f64| -> i32 {
                    for (i, l) in local.iter().enumerate() {
                        if l.sx == sx && l.sy == sy {
                            return i as i32;
                        }
                    }
                    panic!("inconsistent symmetry group");
                };
                let mut induced = vec![Reflection { sx: 1.0, sy: 1.0 }; g0.len()];
                for gi in 0..g0.len() {
                    induced_reflection(&o.place, rg.isx, rg.isy, &g0[gi], &mut induced[gi]);
                }
                let mut covered = vec![false; local.len()];
                let nw = rg.panels.len() as i32;
                for ri in 0..local.len() {
                    if covered[ri] {
                        continue;
                    }
                    for gi in 0..g0.len() {
                        covered[index_of(local[ri].sx * induced[gi].sx, local[ri].sy * induced[gi].sy) as usize] = true;
                    }
                    for n in 0..nw {
                        self.g.sec.push(first + ri as i32 * nw + n);
                        for gi in 0..g0.len() {
                            let image = index_of(local[ri].sx * induced[gi].sx, local[ri].sy * induced[gi].sy);
                            img0[gi].push(first + image * nw + n);
                        }
                    }
                }
            }
        }
        img0
    }

    // The wall images: copies of all panels reflected in the walls (g.walls, the identity first);
    // returns the number of panels before them.
    fn wall_images(&mut self) -> usize {
        let g = &mut self.g;
        g.walls = reflections(self.wx as i32, self.wy as i32);
        let nbase = g.pg.len();
        for wi in 1..g.walls.len() {
            let w = g.walls[wi];
            for j in 0..nbase {
                let v = reflect_vertices(&self.proj[j], &w);
                let p = &g.pg[j];
                let pgw = make_panel_geom(
                    &v,
                    &v3(w.sx * p.n[0], w.sy * p.n[1], p.n[2]),
                    &v3(w.sx * p.c[0], w.sy * p.c[1], p.c[2]),
                    p.area,
                );
                g.pg.push(pgw);
                let rw = reflect_vertices(&g.raw[j], &w);
                g.raw.push(rw);
                self.proj.push(v);
                g.body.push(g.body[j]);
                g.kind.push(g.kind[j]);
                g.lid.push(g.lid[j]);
                g.lid_out.push(-1);
                g.psurf.push(g.psurf[j]);
                g.damper.push(g.damper[j]);
                g.lc.push(g.lc[j]);
                g.press_out.push(-1);
                g.mirror.push(1);
                g.rec.push(-1);
                g.lref.push(g.lref[j]);
                g.nj.push(g.nj[j]); // the image body moves as the mirror image of the body
            }
        }
        nbase
    }

    fn generalized_modes(&mut self) {
        let m = self.m;
        let g = &mut self.g;
        // generalized modes: the displacement at the expanded body-local centroid of every
        // body-surface panel (a wall image is the mirror image of its panel, as for n_j)
        for bi in 0..self.nb {
            for gm in &m.bodies[bi as usize].generalized {
                let mut n = vec![0.0f64; g.Nf as usize];
                let mut w = vec![0.0f64; g.Nf as usize];
                for j in 0..g.Nf {
                    let ju = j as usize;
                    if g.body[ju] != bi || !g.body_surface(j) {
                        continue;
                    }
                    let src = if g.mirror[ju] != 0 { (j % (g.Nf / g.walls.len() as i32)) as usize } else { ju };
                    let d = gm.displacement(&g.lc[src], g.rec[src] + 1);
                    n[ju] = g.nj[src][0] * d[0] + g.nj[src][1] * d[1] + g.nj[src][2] * d[2];
                    w[ju] = d[2];
                }
                g.gn.push(n);
                g.gw.push(w);
            }
        }
    }

    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn solved_characters(&mut self) {
        let (wx, wy) = (self.wx, self.wy);
        let g = &mut self.g;
        // characters; with walls only those even under every wall reflection
        let all = characters(&g.refl);
        for c in all {
            let mut even = true;
            for gi in 0..g.G as usize {
                let e = &g.refl[gi];
                let pure_wall = (if e.sx < 0.0 { wx } else { true })
                    && (if e.sy < 0.0 { wy } else { true })
                    && (e.sx < 0.0 || e.sy < 0.0);
                if pure_wall && c[gi] != 1.0 {
                    even = false;
                }
            }
            if even {
                g.chi.push(c);
            }
        }
    }

    // The field points and the control-surface points in the frame, with their representatives.
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn points(&mut self) {
        let m = self.m;
        let g = &mut self.g;
        let inv = Placement { position: Vec3::ZERO, yaw_deg: -g.yaw };
        // at their length (all exterior without tanks), as the memory estimate counts them
        g.fp.reserve_exact(m.field_points.len());
        for q in 0..m.field_points.len() {
            // the exterior points (the points inside tanks come from their tank problems)
            if m.exterior_point(q) {
                let x = m.field_points[q];
                g.fp.push(inv.rotate(v3(x[0] - g.px, x[1] - g.py, x[2])));
            }
        }
        if m.opt.control_surface_drift {
            g.cs = control_surface_points(m, g.yaw, g.px, g.py);
        }
        g.wavemaker = !m.bodies.is_empty() && (0..m.bodies.len()).all(|b| wall_wavemaker(m, b));
        // the fields at a point g x follow from those at x (phi(g x) = sum_j sigma(g j) S(x, j)), so only
        // one point of every orbit of the frame's reflection group needs its own influence rows; not in a
        // channel, whose wall terms are evaluated per point
        {
            let mut scale: f64 = 1.0;
            for c in &g.cs {
                scale = scale.max(c.x[0].abs()).max(c.x[1].abs()).max(c.x[2].abs());
            }
            let tol = 1e-9 * scale;
            for q in 0..g.cs.len() {
                let x = g.cs[q].x;
                let (mut rep, mut ge) = (-1i32, 0i32);
                if g.W <= 0.0 {
                    let mut r = 0;
                    while r < g.cs_reps.len() && rep < 0 {
                        let y = g.cs[g.cs_reps[r] as usize].x;
                        for gi in 1..g.G as usize {
                            if (g.refl[gi].sx * y[0] - x[0]).abs() <= tol
                                && (g.refl[gi].sy * y[1] - x[1]).abs() <= tol
                                && (y[2] - x[2]).abs() <= tol
                            {
                                rep = r as i32;
                                ge = gi as i32;
                                break;
                            }
                        }
                        r += 1;
                    }
                }
                if rep < 0 {
                    rep = g.cs_reps.len() as i32;
                    g.cs_reps.push(q as i32);
                }
                g.cs_rep.push(rep);
                g.cs_g.push(ge);
            }
        }
    }
}

/// Panels, sectors, images and characters of a problem.
/// m: problem; ints: the interior surface of every body (out).
///
/// Geometry of the problem: a single body in
/// open sea without external lids or fixed dampers in its own horizontal frame with the mesh
/// reflections as the group; else the global frame, the global reflections that
/// map every owner onto itself, and the wall reflections, whose image copies of
/// all panels are added and of whose characters only the even ones are solved.
pub fn build_geometry(m: &Problem, ints: &mut Vec<InteriorPanels>) -> Result<Geometry> {
    let mut b = Builder::new(m, ints);
    b.lid_layout();
    b.dampers_and_pressure_surfaces()?;
    b.body_owners()?;
    b.external_lids();
    b.fixed_dampers();
    b.absorb_lids();
    b.place_panels();
    b.generalized_modes();
    b.solved_characters();
    b.points();
    Ok(b.g)
}
