//! The geometry of the higher-order method: the patch geometry, the
//! frame and the instances of the symmetry group, the quadrature points and blocks, the
//! waterline, the body output points, the field points and the control-surface stencils
//! (theory §7); the compact point arrays of the wave rows and the factored mass matrices of the L2
//! projection, built once with the geometry.

use crate::mesh_desc::PatchKind;
use crate::model_codes::{Domain, RestoringConvention};
use std::collections::BTreeMap;

use crate::bspline::basis;
use crate::constants::PI;
use crate::error::{Error, Result};
use crate::geometry::{Placement, Reflection, Vec3, cross, dot, length_tol, norm, reflections, v3};
use crate::ho_internal::{BPt, FAR_ORDER, HoGeom, Inst, MID_ORDER, PatchDisc, Pt, WlPt, block_of, inst_eval};
use crate::ho_static::rect_size;
use crate::linalg::RealLu;
use crate::lo_geometry::characters;
use crate::lo_geometry::control_surface_points;
use crate::lo_geometry::induced_reflection;
use crate::low_order::{CsPoint, DamperSurf, PressSurf, SurfaceKind, press_shape};
use crate::mesh::{Mesh, patch_eval};
use crate::patch_geometry::PatchGeo;
use crate::plate_geometry::{PlateIndex, Tie};
use crate::polygon::P2;
use crate::problem::{Problem, Role};
use crate::quadrature::gauss_legendre;
use crate::route::wall_wavemaker;

// ------------------------------------------------------------------ patch geometry

/// Geometry of stored patch p of mesh m.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn patch_geo(m: &Mesh, p: i32) -> PatchGeo<'_> {
    let mut g = PatchGeo::default();
    let pu = p as usize;
    if m.trimmed() {
        // an element of a flat patch whose submerged part is a quadrilateral (wholly wet, or
        // every fibre cut) is the flat patch of its four corners (u reversed back to the vertex
        // order of flat patches); else a smooth map with x_u x x_v into the body, as an analytic patch
        let mut quad = m.source.as_ref().unwrap().geometry == PatchKind::Flat;
        if quad && m.trim[pu].cut != 0 {
            let tab = &m.trim_root[pu];
            for &r in tab {
                quad = quad && !r.is_nan();
            }
            quad = quad && !tab.is_empty();
        }
        if quad {
            let c: [[f64; 2]; 4] = [[1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
            for i in 0..4 {
                let (mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO);
                patch_eval(m, p, c[i][0], c[i][1], &mut g.vt[i], &mut xu, &mut xv);
            }
            g.flat = true;
            return g;
        }
        g.flat = false;
        g.mesh = Some(m);
        g.rec = p;
        return g;
    }
    g.flat = m.geometry == PatchKind::Flat;
    if g.flat {
        let v = &m.vertices[12 * pu..];
        for i in 0..4 {
            g.vt[i] = v3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        }
        return g;
    }
    // a B-spline or analytic patch: evaluated by patch_eval of its mesh
    g.analytic = m.geometry == PatchKind::Analytic;
    g.mesh = Some(m);
    g.rec = p;
    g
}

/// Open uniform knots of n spans and order k on [-1, 1].
pub fn open_knots(n: i32, k: i32) -> Vec<f64> {
    let k = k.max(0) as usize;
    let mut t = vec![-1.0; k];
    t.extend((1..n).map(|i| -1.0 + 2.0 * i as f64 / n as f64));
    t.resize(t.len() + k, 1.0);
    t
}

/// Generalized normals of a point from its body-local position and normal (the
/// rigid modes of its body in body axes, the shapes of its pressure modes,
/// interior 0); a wall image has the values of its original (mirror motion). nm: NM values.
pub fn point_modes(h: &HoGeom, ii: i32, xb: &Vec3, nb: &Vec3, nm: &mut [f64]) {
    nm[..h.NM as usize].fill(0.0);
    let kd = h.kind_of(ii);
    let b = h.inst[ii as usize].body;
    if kd == SurfaceKind::Wet || kd == SurfaceKind::Sheet || kd == SurfaceKind::Plate {
        let r = *xb - h.rc[b as usize];
        let mom = cross(r, *nb);
        let v = [nb[0], nb[1], nb[2], mom[0], mom[1], mom[2]];
        for k in 0..6 {
            nm[(6 * b) as usize + k] = v[k];
        }
        for k in 0..h.gmode.len() {
            // generalized modes
            if h.gm_body[k] == b {
                let d = h.gmode[k].displacement(xb, h.prec[h.inst[ii as usize].patch as usize]);
                nm[(6 * h.NB) as usize + k] = nb[0] * d[0] + nb[1] * d[1] + nb[2] * d[2];
            }
        }
    } else if kd == SurfaceKind::Press {
        let s = h.pss[h.inst[ii as usize].patch as usize];
        for k in 0..(h.NM - h.P0) as usize {
            if h.pm_ps[k] == s {
                nm[h.P0 as usize + k] = press_shape(h.pm_shape[k], &v3(xb[0], xb[1], 0.0));
            }
        }
    }
}

/// The quadrature point at (u, v) of panel (iu, iv) of instance ii: position, normal, weight
/// (times the area element when area), basis functions and generalized normals.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn fill_point(h: &HoGeom, ii: i32, iu: i32, iv: i32, u: f64, v: f64, w_par: f64, area: bool, p: &mut Pt) {
    let in_ = &h.inst[ii as usize];
    let d = &h.disc[in_.patch as usize];
    let mut J = 0.0;
    let mut nb = Vec3::ZERO;
    inst_eval(h, in_, u, v, &mut p.x, &mut p.n, &mut J, Some(&mut p.xl), Some(&mut nb));
    p.w = if area { w_par * J } else { w_par };
    p.inst = ii;
    p.iu = iu;
    p.iv = iv;
    p.uf = u;
    p.vf = v;
    basis(&d.Tu, d.ku, iu + d.ku - 1, u, &mut p.bu, None);
    basis(&d.Tv, d.kv, iv + d.kv - 1, v, &mut p.bv, None);
    let xl = p.xl;
    point_modes(h, ii, &xl, &nb, &mut p.nm);
}

// The patches of every body in the exterior problem: not those of its internal tanks, which
// are problems of their own; keep the stored elements of each body, pfirst the
// first patch of each body.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn add_patches<'a>(m: &'a Problem, h: &mut HoGeom<'a>, keep: &mut [Vec<i32>], pfirst: &mut Vec<i32>) {
    for b in 0..h.NB as usize {
        let mesh = &m.meshes[m.bodies[b].mesh as usize];
        // the patches that are neither a tank wall nor inside a tank (a tank is its own problem)
        for p in 0..mesh.count as usize {
            if m.derived()[b].role[p] != Role::Tank && m.derived()[b].tank[p] < 0 {
                keep[b].push(p as i32);
            }
        }
    }
    for b in 0..h.NB as usize {
        let mesh = &m.meshes[m.bodies[b].mesh as usize];
        pfirst.push(h.Np);
        for &p in &keep[b] {
            h.geo.push(patch_geo(mesh, p));
            let dd = &mesh.discretization[p as usize];
            let d = PatchDisc {
                nu: dd.nu,
                nv: dd.nv,
                ku: dd.ku,
                kv: dd.kv,
                iquo: dd.iquo,
                iqui: dd.iqui,
                Tu: open_knots(dd.nu, dd.ku),
                Tv: open_knots(dd.nv, dd.kv),
            };
            h.disc.push(d);
            h.pbody.push(b as i32);
            h.prec.push(p + 1);
        }
        for g in &m.bodies[b].generalized {
            h.gmode.push(g);
            h.gm_body.push(b as i32);
        }
        h.Np += keep[b].len() as i32;
        h.rc.push(m.bodies[b].rotation_center);
    }
}

// The kinds of the patches and the pressure surfaces (coordinates after the rigid ones of their body).
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn patch_kinds(m: &Problem, h: &mut HoGeom, keep: &[Vec<i32>], pfirst: &[i32]) -> Result<()> {
    let np = h.Np as usize;
    h.kind = vec![SurfaceKind::Wet; np];
    h.pss = vec![-1; np];
    h.pdmp = vec![-1; np];
    h.lidx = vec![-1; np];
    for b in 0..h.NB as usize {
        let body = &m.bodies[b];
        let mesh = &m.meshes[body.mesh as usize];
        let der = &m.derived()[b];
        let mut coord = der.coordinate_offset + 6 + der.generalized;
        let s0 = h.ps.len() as i32;
        let d0 = h.dmp.len() as i32;
        for dm in &body.dampers {
            h.dmp.push(DamperSurf { eps: dm.epsilon, free: dm.free(), porous: dm.porous() });
        }
        for s in 0..body.pressure.len() {
            let mut q = PressSurf { body: b as i32, coord0: coord, id: s as i32 + 1, ..PressSurf::default() };
            for pm in &body.pressure[s].modes {
                q.shapes.push(pm.shape);
                h.pm_ps.push(s0 + s as i32);
                h.pm_shape.push(pm.shape);
            }
            q.chamber = body.pressure[s].chamber_coupling;
            q.numeric = body.pressure[s].convention == RestoringConvention::NumericCompatibility;
            h.ps.push(q);
            coord += body.pressure[s].modes.len() as i32;
        }
        for k in 0..keep[b].len() {
            let p = keep[b][k] as usize;
            let q = (pfirst[b] + k as i32) as usize;
            let role = der.role[p];
            if role == Role::Interior {
                h.kind[q] = SurfaceKind::Interior;
            }
            if role == Role::Dipole {
                // a planar dipole patch
                h.kind[q] = SurfaceKind::Sheet;
                h.sheets = true;
                let d = der.owner[p]; // a dipole damper
                if d >= 0 {
                    h.pdmp[q] = d0 + d;
                }
            }
            if role == Role::Lid {
                // a damping lid as patches of the body mesh
                h.kind[q] = SurfaceKind::Lid;
                h.lidx[q] = der.owner[p];
            }
            if role == Role::Pressure {
                h.kind[q] = SurfaceKind::Press;
                h.pss[q] = s0 + der.owner[p];
            }
            if role == Role::Plate {
                // a free-surface plate as patches of the body mesh (those of its own mesh joined)
                h.kind[q] = SurfaceKind::Plate;
            }
        }
        // a pressure surface in z = 0: the centres of all its patches in the plane (the plane test
        // of theory §14); one with some patches in it and some not is refused
        let tol = length_tol(mesh.reference_length);
        for s in s0 as usize..h.ps.len() {
            let (mut inside, mut outside) = (false, false);
            for k in 0..keep[b].len() {
                if h.pss[(pfirst[b] + k as i32) as usize] == s as i32 {
                    let sh = &mesh.shapes[keep[b][k] as usize];
                    let z0 = (sh.centre[2] + body.position[2]).abs() <= tol;
                    inside |= z0;
                    outside |= !z0;
                }
            }
            if inside && outside {
                return Err(Error::value(format!(
                    "body '{}': pressure surface {}: some of its patches lie in z = 0 and some do not (a \
                     pressure surface lies in the free surface or below it)",
                    body.name,
                    s - s0 as usize + 1
                )));
            }
            h.ps[s].z0 = inside;
        }
    }
    h.P0 = 6 * h.NB + h.gmode.len() as i32;
    h.NM = h.P0 + h.pm_ps.len() as i32;
    Ok(())
}

// The frame and the group: a single body in open sea in its
// horizontal frame with its mesh reflections; else the global frame, the global
// reflections that map every body onto itself and the wall images. Then the instances
// of every patch and the row instances.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn frame_and_instances(m: &Problem, h: &mut HoGeom, keep: &[Vec<i32>], pfirst: &[i32]) {
    let wx = m.opt.domain == Domain::XWall || m.opt.domain == Domain::OrthogonalWalls;
    let wy = m.opt.domain == Domain::YWall || m.opt.domain == Domain::OrthogonalWalls;
    let channel = m.opt.domain == Domain::Channel;
    let single = h.NB == 1 && !wx && !wy && !channel;
    h.single = single;
    if single {
        let b = &m.bodies[0];
        let mesh = &m.meshes[b.mesh as usize];
        h.px = b.position[0];
        h.py = b.position[1];
        h.yaw = b.yaw;
        h.refl = reflections(mesh.isx, mesh.isy);
        h.ind = vec![h.refl.clone()];
    } else {
        let (mut gx, mut gy) = (!wx, !wy);
        for b in &m.bodies {
            let mesh = &m.meshes[b.mesh as usize];
            let mut l = Reflection { sx: 1.0, sy: 1.0 };
            gx = gx && induced_reflection(b, mesh.isx, mesh.isy, &Reflection { sx: -1.0, sy: 1.0 }, &mut l);
            gy = gy && induced_reflection(b, mesh.isx, mesh.isy, &Reflection { sx: 1.0, sy: -1.0 }, &mut l);
        }
        h.refl = reflections(gx as i32, gy as i32);
        for b in &m.bodies {
            let mesh = &m.meshes[b.mesh as usize];
            let mut ind = vec![Reflection { sx: 1.0, sy: 1.0 }; h.refl.len()];
            for gi in 0..h.refl.len() {
                induced_reflection(b, mesh.isx, mesh.isy, &h.refl[gi], &mut ind[gi]);
            }
            h.ind.push(ind);
        }
    }
    h.G = h.refl.len() as i32;
    h.chi = characters(&h.refl);
    h.walls = reflections(wx as i32, wy as i32);
    for b in 0..h.NB as usize {
        let body = &m.bodies[b];
        let mesh = &m.meshes[body.mesh as usize];
        let local = reflections(mesh.isx, mesh.isy);
        let yaw = if single { 0.0 } else { body.yaw * PI / 180.0 };
        let t = v3(
            if single { 0.0 } else { body.position[0] },
            if single { 0.0 } else { body.position[1] },
            body.position[2],
        );
        let first = h.inst.len() as i32;
        let nk = keep[b].len() as i32;
        for r in &local {
            for p in 0..nk {
                let in_ = Inst {
                    patch: pfirst[b] + p,
                    r: *r,
                    body: b as i32,
                    c: yaw.cos(),
                    s: yaw.sin(),
                    t,
                    ..Inst::default()
                };
                h.inst.push(in_);
            }
        }
        let index_of = |sx: f64, sy: f64| -> i32 {
            for i in 0..local.len() {
                if local[i].sx == sx && local[i].sy == sy {
                    return i as i32;
                }
            }
            unreachable!("the group holds the local reflection ({sx}, {sy}) of every instance");
        };
        // row instances: one local reflection per orbit of the group
        let mut covered = vec![false; local.len()];
        for ri in 0..local.len() {
            if covered[ri] {
                continue;
            }
            for gi in 0..h.G {
                let d = h.ind[b][gi as usize];
                let li = index_of(local[ri].sx * d.sx, local[ri].sy * d.sy);
                covered[li as usize] = true;
                for p in 0..nk {
                    let in_ = &mut h.inst[(first + li * nk + p) as usize];
                    in_.rep = first + ri as i32 * nk + p;
                    in_.g = gi;
                }
            }
            for p in 0..nk {
                h.rows.push(first + ri as i32 * nk + p);
            }
        }
    }
}

// The wall images of every instance (the same unknowns); wavemakers in a wall: not
// the images in their own wall plane, which coincide with a kept instance (the factor 2 of
// their sources is applied in the equations and the field).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn wall_images(m: &Problem, h: &mut HoGeom) {
    h.wavemaker = !m.bodies.is_empty() && (0..m.bodies.len()).all(|b| wall_wavemaker(m, b));
    let in_plane = |h: &HoGeom, in_: &Inst, axis: usize| -> bool {
        let (mut x, mut n) = (Vec3::ZERO, Vec3::ZERO);
        let mut J = 0.0;
        for u in [-1.0, 0.0, 1.0] {
            for v in [-1.0, 0.0, 1.0] {
                inst_eval(h, in_, u, v, &mut x, &mut n, &mut J, None, None);
                if x[axis].abs() > length_tol(m.meshes[m.bodies[in_.body as usize].mesh as usize].reference_length) {
                    return false;
                }
            }
        }
        true
    };
    let nbase = h.inst.len();
    for wi in 1..h.walls.len() {
        for i in 0..nbase {
            let base = h.inst[i];
            if h.wavemaker
                && ((h.walls[wi].sx < 0.0 && in_plane(h, &base, 0)) || (h.walls[wi].sy < 0.0 && in_plane(h, &base, 1)))
            {
                continue;
            }
            let mut in_ = base;
            in_.w = h.walls[wi];
            in_.image = true;
            h.inst.push(in_);
        }
    }
}

// The near images of a channel (theory §8.5): every instance reflected in the walls y = W/2 and
// y = -W/2 (y -> W - y, y -> -W - y), with the unknowns of their original and the mirror motion
// as the wall images, their columns scaled by the reflection factor; the farther images are
// summed at the points of the wave rows (kernel_row).
fn channel_images(m: &Problem, h: &mut HoGeom) {
    if m.opt.domain != Domain::Channel {
        return;
    }
    h.W = m.opt.channel_width;
    let nbase = h.inst.len();
    for wt in [h.W, -h.W] {
        for i in 0..nbase {
            let mut in_ = h.inst[i];
            in_.w = Reflection { sx: 1.0, sy: -1.0 };
            in_.wt = wt;
            in_.f = m.opt.channel_reflect;
            in_.image = true;
            h.inst.push(in_);
        }
    }
}

// The unknowns of the row instances, the columns and the panel blocks of every instance.
fn layout(m: &Problem, h: &mut HoGeom) {
    h.dof0 = vec![0; h.inst.len()];
    for k in 0..h.rows.len() {
        let ii = h.rows[k];
        h.dof0[ii as usize] = h.Ndof;
        h.Ndof += h.disc_of(ii).nb();
    }
    for i in 0..h.inst.len() {
        h.dof0[i] = h.dof0[h.inst[i].rep as usize];
    }
    let mut nblk = 0;
    for i in 0..h.inst.len() {
        h.col0.push(h.Ncol);
        h.Ncol += h.disc_of(i as i32).nb();
        h.blk0.push(nblk);
        nblk += h.panels_of(h.inst[i].patch);
    }
    h.depth = m.env.depth;
}

// The outer points on the row instances and the inner points on every instance, with their
// output indices and basis columns.
fn quadrature_points(h: &mut HoGeom) {
    // outer points on the row instances
    for k in 0..h.rows.len() {
        let p = h.rows[k];
        let (nu, nv, iquo) = {
            let d = h.disc_of(p);
            (d.nu, d.nv, d.iquo)
        };
        let (gx, gw) = gauss_legendre(iquo);
        let (du, dv) = (1.0 / nu as f64, 1.0 / nv as f64); // half widths of the panels
        for iv in 0..nv {
            for iu in 0..nu {
                for a in 0..iquo as usize {
                    for c in 0..iquo as usize {
                        let u = -1.0 + du * ((2 * iu + 1) as f64 + gx[a]);
                        let v = -1.0 + dv * ((2 * iv + 1) as f64 + gx[c]);
                        let mut pt = Pt::default();
                        fill_point(h, p, iu, iv, u, v, gw[a] * gw[c] * du * dv, false, &mut pt);
                        h.outer.push(pt);
                    }
                }
            }
        }
    }
    // inner points on every instance, per panel
    for ii in 0..h.inst.len() {
        let (nu, nv, iqui) = {
            let d = &h.disc[h.inst[ii].patch as usize];
            (d.nu, d.nv, d.iqui)
        };
        let (gx, gw) = gauss_legendre(iqui);
        let (du, dv) = (1.0 / nu as f64, 1.0 / nv as f64); // half widths of the panels
        for iv in 0..nv {
            for iu in 0..nu {
                h.inner0.push(h.inner.len() as i32);
                for a in 0..iqui as usize {
                    for c in 0..iqui as usize {
                        let u = -1.0 + du * ((2 * iu + 1) as f64 + gx[a]);
                        let v = -1.0 + dv * ((2 * iv + 1) as f64 + gx[c]);
                        let mut pt = Pt::default();
                        fill_point(h, ii as i32, iu, iv, u, v, gw[a] * gw[c] * du * dv, true, &mut pt);
                        h.inner.push(pt);
                    }
                }
            }
        }
    }
    h.inner0.push(h.inner.len() as i32);
    for i in 0..h.inner.len() {
        let qi = h.inner[i].inst;
        let o = if h.kind_of(qi) == SurfaceKind::Press && !h.inst[qi as usize].image {
            h.Q += 1;
            h.Q - 1
        } else {
            -1
        };
        h.press_out.push(o);
    }
    for i in 0..h.inner.len() {
        let qi = h.inner[i].inst;
        let o = if h.kind_of(qi) == SurfaceKind::Lid && !h.inst[qi as usize].image {
            h.L += 1;
            h.L - 1
        } else {
            -1
        };
        h.lid_out.push(o);
    }
    for i in 0..h.inner.len() {
        let qi = h.inner[i].inst as usize;
        let (ku, kv, mu) = {
            let dc = &h.disc[h.inst[qi].patch as usize];
            (dc.ku, dc.kv, dc.mu())
        };
        let c0 = h.col0[qi];
        let q = &mut h.inner[i];
        q.nk = 0;
        for jb in 0..kv {
            for ja in 0..ku {
                let nk = q.nk as usize;
                q.col[nk] = c0 + (q.iv + jb) * mu + q.iu + ja;
                q.wb[nk] = q.w * q.bu[ja as usize] * q.bv[jb as usize];
                q.nk += 1;
            }
        }
    }
}

// The mean waterline: the sides of the wet patches in z = 0, 6 Gauss points per panel.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn waterline_points(h: &mut HoGeom) {
    // the validation hook of the sensitivity studies (6 without it)
    let nq = (crate::env_hooks::validation_constant("WATERLINE_GAUSS") as i32).clamp(1, 64);
    let (gx, gw) = gauss_legendre(nq);
    // the free-surface plates (not the wall images) as horizontal polygons, 16 points a side, with
    // their bodies and their longest sides: a waterline point whose water beside it lies under a
    // plate (the plate's edge on its wall) takes the band under the plate in the pressure drift
    // (theory §11.2); the point a twentieth of the panel side into the water within a fortieth of
    // the side or a tenth of the plate's longest side of a plate (the chords of two meshes of one
    // curve: the rule of the low order and of the route's walls), the nearest one
    let plates: Vec<(i32, Vec<P2>, f64)> = h
        .inst
        .iter()
        .filter(|q| !q.image && h.kind[q.patch as usize] == SurfaceKind::Plate)
        .map(|q| {
            let s = |k: usize| -1.0 + 2.0 * k as f64 / 16.0;
            let uv: Vec<(f64, f64)> = (0..16)
                .map(|k| (s(k), -1.0))
                .chain((0..16).map(|k| (1.0, s(k))))
                .chain((0..16).map(|k| (-s(k), 1.0)))
                .chain((0..16).map(|k| (-1.0, -s(k))))
                .collect();
            let poly: Vec<P2> = uv
                .iter()
                .map(|&(u, v)| {
                    let (mut x, mut n) = (Vec3::ZERO, Vec3::ZERO);
                    let mut J = 0.0;
                    inst_eval(h, q, u, v, &mut x, &mut n, &mut J, None, None);
                    [x[0], x[1]]
                })
                .collect();
            // the length of each side of the patch (its 16 chords)
            let side = (0..4)
                .map(|s| {
                    (0..16)
                        .map(|k| {
                            let (a, c) = (poly[16 * s + k], poly[(16 * s + k + 1) % 64]);
                            (c[0] - a[0]).hypot(c[1] - a[1])
                        })
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            (q.body, poly, side)
        })
        .collect();
    let plates = PlateIndex::new(plates);
    // the nearest plate (the last of equal distance, in the order of the instances)
    let plate_at = |x: &Vec3, n: &Vec3, seg: f64| -> i32 {
        plates.beside([x[0], x[1]], [n[0], n[1]], seg, Tie::Last).map_or(-1, |i| plates.elements[i].0)
    };
    for ii in 0..h.inst.len() {
        let in_ = h.inst[ii];
        if in_.image || h.kind[in_.patch as usize] != SurfaceKind::Wet {
            continue;
        }
        let (nu, nv) = {
            let d = &h.disc[in_.patch as usize];
            (d.nu, d.nv)
        };
        for side in 0..4 {
            // 0: v = -1, 1: v = +1 (u varies); 2: u = -1, 3: u = +1 (v varies)
            let along_u = side < 2;
            let fixed = if side % 2 == 0 { -1.0 } else { 1.0 };
            let at = |h: &HoGeom, t: f64| -> Vec3 {
                let (mut x, mut n) = (Vec3::ZERO, Vec3::ZERO);
                let mut J = 0.0;
                inst_eval(
                    h,
                    &in_,
                    if along_u { t } else { fixed },
                    if along_u { fixed } else { t },
                    &mut x,
                    &mut n,
                    &mut J,
                    None,
                    None,
                );
                x
            };
            let a = at(h, -1.0);
            let c = at(h, 0.0);
            let e = at(h, 1.0);
            let tol = length_tol(norm(e - a));
            if a[2].abs() > tol || c[2].abs() > tol || e[2].abs() > tol {
                continue;
            }
            let np = if along_u { nu } else { nv };
            for k in 0..np {
                for gq in 0..nq as usize {
                    let half = 1.0 / np as f64;
                    let t = -1.0 + half * ((2 * k + 1) as f64 + gx[gq]);
                    let mut p = WlPt {
                        inst: ii as i32,
                        u: if along_u { t } else { fixed },
                        v: if along_u { fixed } else { t },
                        iu: if along_u {
                            k
                        } else if side == 2 {
                            0
                        } else {
                            nu - 1
                        },
                        iv: if along_u { if side == 0 { 0 } else { nv - 1 } } else { k },
                        ..WlPt::default()
                    };
                    let mut J = 0.0;
                    inst_eval(h, &in_, p.u, p.v, &mut p.x, &mut p.n, &mut J, Some(&mut p.xb), Some(&mut p.nb));
                    let (mut xl, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                    h.geo[in_.patch as usize].eval(p.u, p.v, &mut xl, &mut xu, &mut xv);
                    p.dl = norm(if along_u { xu } else { xv }) * gw[gq] * half;
                    p.plate = plate_at(&p.x, &p.n, norm(e - a) / np as f64);
                    h.wl.push(p);
                }
            }
        }
    }
}

// The body output points: per body and reflection, patches in stored order.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn body_output_points(h: &mut HoGeom) {
    let mut counter: BTreeMap<[i32; 3], i32> = BTreeMap::new(); // (body, sx, sy) -> points so far
    for pt in 0..h.disc.len() {
        for ii in 0..h.inst.len() {
            let in_ = h.inst[ii];
            if in_.patch != pt as i32
                || in_.image
                || (h.kind[pt] != SurfaceKind::Wet
                    && h.kind[pt] != SurfaceKind::Press
                    && h.kind[pt] != SurfaceKind::Plate)
            {
                continue;
            }
            let (dnu, dnv, dku, dkv) = {
                let d = &h.disc[pt];
                (d.nu, d.nv, d.ku, d.kv)
            };
            let key = [in_.body, in_.r.sx as i32, in_.r.sy as i32];
            let mut cnt = *counter.entry(key).or_insert(0);
            let nu = dku + 1;
            let nv = dkv + 1;
            let du = 2.0 / (dnu * nu) as f64;
            let dv = 2.0 / (dnv * nv) as f64;
            for iv in 0..dnv {
                for iu in 0..dnu {
                    for b in 0..nv {
                        for a in 0..nu {
                            // flat patches in the order of the WAMIT-format .5 files: u from
                            // the second vertex towards the first
                            let rev = h.geo[pt].flat;
                            let mut q = BPt {
                                inst: ii as i32,
                                iu: if rev { dnu - 1 - iu } else { iu },
                                iv,
                                body: in_.body,
                                u: (if rev { -1.0 } else { 1.0 }) * (-1.0 + du * ((iu * nu + a) as f64 + 0.5)),
                                v: -1.0 + dv * ((iv * nv + b) as f64 + 0.5),
                                ..BPt::default()
                            };
                            let mut J = 0.0;
                            let (mut xb, mut nb) = (Vec3::ZERO, Vec3::ZERO);
                            inst_eval(h, &in_, q.u, q.v, &mut q.x, &mut q.n, &mut J, Some(&mut xb), Some(&mut nb));
                            q.area = J * du * dv;
                            point_modes(h, ii as i32, &xb, &nb, &mut q.nm);
                            cnt += 1;
                            q.rec = cnt;
                            h.bpts.push(q);
                        }
                    }
                }
            }
            counter.insert(key, cnt);
        }
    }
}

// The specified body points (the item body_points): the closest point of the body's wetted
// and pressure patches (not wall images), body-local; coarse 8 x 8 parametric
// grid per instance, then Gauss-Newton in (u, v) clamped to [-1, 1].
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn specified_body_points(m: &Problem, h: &mut HoGeom) -> Result<()> {
    h.bpts.clear();
    let local = |h: &HoGeom, in_: &Inst, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3| {
        let (mut a0, mut au, mut av) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        h.geo[in_.patch as usize].eval(u, v, &mut a0, &mut au, &mut av);
        *x = v3(in_.r.sx * a0[0], in_.r.sy * a0[1], a0[2]);
        *xu = v3(in_.r.sx * au[0], in_.r.sy * au[1], au[2]);
        *xv = v3(in_.r.sx * av[0], in_.r.sy * av[1], av[2]);
    };
    for bi in 0..h.NB {
        for i in 0..m.bodies[bi as usize].body_points.len() {
            let pt = m.bodies[bi as usize].body_points[i];
            let (mut best, mut bu, mut bv) = (f64::INFINITY, 0.0, 0.0);
            let mut bin = -1i32;
            for ii in 0..h.inst.len() {
                let in_ = h.inst[ii];
                if in_.body != bi
                    || in_.image
                    || (h.kind[in_.patch as usize] != SurfaceKind::Wet
                        && h.kind[in_.patch as usize] != SurfaceKind::Press
                        && h.kind[in_.patch as usize] != SurfaceKind::Plate)
                {
                    continue;
                }
                let (mut u, mut v, mut d0) = (0.0, 0.0, f64::INFINITY);
                for a1 in 0..8 {
                    for b1 in 0..8 {
                        let uu = -1.0 + (a1 as f64 + 0.5) / 4.0;
                        let vv = -1.0 + (b1 as f64 + 0.5) / 4.0;
                        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                        local(h, &in_, uu, vv, &mut x, &mut xu, &mut xv);
                        let dd = norm(x - pt);
                        if dd < d0 {
                            d0 = dd;
                            u = uu;
                            v = vv;
                        }
                    }
                }
                for _it in 0..40 {
                    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                    local(h, &in_, u, v, &mut x, &mut xu, &mut xv);
                    let r = x - pt;
                    let a11 = dot(xu, xu);
                    let a12 = dot(xu, xv);
                    let a22 = dot(xv, xv);
                    let g1 = dot(xu, r);
                    let g2 = dot(xv, r);
                    let det = a11 * a22 - a12 * a12;
                    if !(det > 0.0) {
                        break;
                    }
                    let du = -(a22 * g1 - a12 * g2) / det;
                    let dv = -(a11 * g2 - a12 * g1) / det;
                    let un = (u + du).clamp(-1.0, 1.0);
                    let vn = (v + dv).clamp(-1.0, 1.0);
                    let done = (un - u).abs() + (vn - v).abs() < 1e-13;
                    u = un;
                    v = vn;
                    if done {
                        break;
                    }
                }
                let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                local(h, &in_, u, v, &mut x, &mut xu, &mut xv);
                let dd = norm(x - pt);
                if dd < best {
                    best = dd;
                    bu = u;
                    bv = v;
                    bin = ii as i32;
                }
            }
            if bin < 0 {
                return Err(Error::internal("no wetted patch for the body points"));
            }
            let in_ = h.inst[bin as usize];
            let (dnu, dnv) = {
                let d = &h.disc[in_.patch as usize];
                (d.nu, d.nv)
            };
            let mut q = BPt {
                inst: bin,
                given: true,
                body: bi,
                rec: i as i32 + 1,
                u: bu,
                v: bv,
                iu: (dnu - 1).min(((bu + 1.0) * 0.5 * dnu as f64) as i32),
                iv: (dnv - 1).min(((bv + 1.0) * 0.5 * dnv as f64) as i32),
                ..BPt::default()
            };
            let mut J = 0.0;
            let (mut xb, mut nb) = (Vec3::ZERO, Vec3::ZERO);
            inst_eval(h, &in_, bu, bv, &mut q.x, &mut q.n, &mut J, Some(&mut xb), Some(&mut nb));
            q.area = 0.0;
            point_modes(h, bin, &xb, &nb, &mut q.nm);
            h.bpts.push(q);
        }
    }
    Ok(())
}

/// Whether a stencil point at height z (step s) lies in z = 0 and takes the one-sided stencil.
/// push_stencil and the two difference formulas of ho_frequency.rs call it, so that the layout
/// of the points and the formula read from them agree.
#[inline]
pub(crate) fn stencil_top(z: f64, s: f64) -> bool {
    z.abs() <= 1e-9 * s
}

/// The six difference points of the velocity stencil of x with step s: x +- s e_x,
/// x +- s e_y and x +- s e_z, one-sided (z, z - s, z - 2s) for a point in z = 0.
fn push_stencil(fp: &mut Vec<Vec3>, x: Vec3, s: f64) {
    let top = stencil_top(x[2], s);
    fp.push(v3(x[0] + s, x[1], x[2]));
    fp.push(v3(x[0] - s, x[1], x[2]));
    fp.push(v3(x[0], x[1] + s, x[2]));
    fp.push(v3(x[0], x[1] - s, x[2]));
    fp.push(v3(x[0], x[1], if top { x[2] - s } else { x[2] + s }));
    fp.push(v3(x[0], x[1], if top { x[2] - 2.0 * s } else { x[2] - s }));
}

// The exterior field points in the frame and, for the field velocity, their difference stencils.
fn field_points(m: &Problem, h: &mut HoGeom) {
    let inv = Placement { position: v3(0.0, 0.0, 0.0), yaw_deg: -h.yaw };
    // at their length with the stencils (all exterior without tanks), as the memory estimate counts
    // them, not at the doubled capacity of their pushes
    h.fp.reserve_exact(m.field_points.len() * if m.opt.field_velocity { 7 } else { 1 });
    for q in 0..m.field_points.len() {
        // the exterior points (the points inside a tank: by the radiation problem of that tank)
        if m.exterior_point(q) {
            let x = m.field_points[q];
            h.fp.push(inv.rotate(v3(x[0] - h.px, x[1] - h.py, x[2])));
        }
    }
    h.Pu = h.fp.len() as i32;
    if m.opt.field_velocity {
        // the velocity by differences of the potential: x +- s e_x, x +- s e_y and
        // x +- s e_z, one-sided (z, z - s, z - 2s) for a point in z = 0; s = 1e-4 L
        h.fstep = 1e-4 * m.meshes[m.bodies[0].mesh as usize].reference_length;
        let s = h.fstep;
        for q in 0..h.Pu as usize {
            let x = h.fp[q];
            push_stencil(&mut h.fp, x, s);
        }
    }
}

// The control-surface points and their velocity stencils as for field points.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn control_surface_stencils(m: &Problem, h: &mut HoGeom) {
    h.cs = control_surface_points(m, h.yaw, h.px, h.py);
    // one body without walls: the field at g x is that at x with the solution of the instance
    // reflected by g (phi(g x) = sum_j D_j(x) c_{g j}), so one point of every orbit of the
    // frame's group is evaluated; the instance with the local reflection g r of each instance
    if h.NB == 1 && h.walls.len() == 1 && h.G > 1 {
        let Ni = h.inst.len();
        let mut col = vec![vec![-1i32; h.Ncol as usize]; h.G as usize];
        let mut ok = true;
        let mut gi = 0;
        while gi < h.G as usize && ok {
            let mut ii = 0;
            while ii < Ni && ok {
                let a = &h.inst[ii];
                let mut jj = -1i32;
                let mut k = 0;
                while k < Ni && jj < 0 {
                    let b = &h.inst[k];
                    if b.patch == a.patch
                        && b.body == a.body
                        && b.image == a.image
                        && b.wt == h.refl[gi].sy * a.wt
                        && b.r.sx == h.refl[gi].sx * a.r.sx
                        && b.r.sy == h.refl[gi].sy * a.r.sy
                    {
                        jj = k as i32;
                    }
                    k += 1;
                }
                if jj < 0 {
                    ok = false;
                } else {
                    for l in 0..h.disc[a.patch as usize].nb() {
                        col[gi][(h.col0[ii] + l) as usize] = h.col0[jj as usize] + l;
                    }
                }
                ii += 1;
            }
            gi += 1;
        }
        if ok {
            // per group element the instance whose columns start at the reflected column of the
            // first column of each instance (usize::MAX: an instance without columns)
            h.cs_inst = col
                .iter()
                .map(|cg| {
                    (0..Ni)
                        .map(|ii| {
                            if h.disc[h.inst[ii].patch as usize].nb() == 0 {
                                return usize::MAX;
                            }
                            let c = cg[h.col0[ii] as usize];
                            h.col0.iter().position(|&c0| c0 == c).expect("a reflected instance")
                        })
                        .collect()
                })
                .collect();
            h.cs_G = h.G;
            h.cs_col = col;
        }
    }
    {
        let mut scale = 1.0f64;
        for c in &h.cs {
            scale = scale.max(c.x[0].abs()).max(c.x[1].abs()).max(c.x[2].abs());
        }
        let tol = 1e-9 * scale;
        for ci in 0..h.cs.len() {
            let c: &CsPoint = &h.cs[ci];
            let (mut rep, mut ge) = (-1i32, 0i32);
            let mut r = 0;
            while r < h.cs_reps.len() && rep < 0 {
                let y = h.cs[h.cs_reps[r] as usize].x;
                for gi in 1..h.cs_G as usize {
                    if (h.refl[gi].sx * y[0] - c.x[0]).abs() <= tol
                        && (h.refl[gi].sy * y[1] - c.x[1]).abs() <= tol
                        && (y[2] - c.x[2]).abs() <= tol
                    {
                        rep = r as i32;
                        ge = gi as i32;
                        break;
                    }
                }
                r += 1;
            }
            if rep < 0 {
                rep = h.cs_reps.len() as i32;
                h.cs_reps.push(ci as i32);
            }
            h.cs_rep.push(rep);
            h.cs_g.push(ge);
        }
    }
    h.cs0 = h.fp.len() as i32;
    h.csstep = 1e-4 * m.meshes[m.bodies[0].mesh as usize].reference_length;
    let s = h.csstep;
    h.fp.reserve_exact(7 * h.cs_reps.len());
    for k in 0..h.cs_reps.len() {
        let x = h.cs[h.cs_reps[k] as usize].x;
        h.fp.push(x);
    }
    for k in 0..h.cs_reps.len() {
        let x = h.cs[h.cs_reps[k] as usize].x;
        push_stencil(&mut h.fp, x, s);
    }
}

/// The geometry, instances, quadrature points and outputs of the problem.
pub fn build(m: &Problem) -> Result<HoGeom<'_>> {
    let mut h = HoGeom { NB: m.bodies.len() as i32, ..HoGeom::default() };
    let mut pfirst: Vec<i32> = Vec::new(); // first stored patch of each body
    let mut keep: Vec<Vec<i32>> = vec![Vec::new(); h.NB as usize];
    add_patches(m, &mut h, &mut keep, &mut pfirst);
    patch_kinds(m, &mut h, &keep, &pfirst)?;
    frame_and_instances(m, &mut h, &keep, &pfirst);
    wall_images(m, &mut h);
    channel_images(m, &mut h);
    layout(m, &mut h);
    quadrature_points(&mut h);
    waterline_points(&mut h);
    body_output_points(&mut h);
    if !m.bodies.is_empty() && !m.bodies[0].body_points.is_empty() {
        specified_body_points(m, &mut h)?;
    }
    let nb = h.inner0.len() - 1;
    h.bc.resize(nb, Vec3::ZERO);
    h.bs.resize(nb, 0.0);
    field_points(m, &mut h);
    if m.opt.control_surface_drift {
        control_surface_stencils(m, &mut h);
    }
    Ok(h)
}

/// The inner and mid-tier point blocks of every (instance, panel).
pub fn build_blocks(h: &mut HoGeom) {
    let (gx, gw) = gauss_legendre(MID_ORDER);
    let (fx, fw) = gauss_legendre(FAR_ORDER);
    let (mo, fo) = (MID_ORDER as usize, FAR_ORDER as usize);
    // the blocks are filled outside h (fill_point and rect_size read the rest of h)
    let mut mid = std::mem::take(&mut h.mid);
    let mut sfar = std::mem::take(&mut h.sfar);
    let mut bs = std::mem::take(&mut h.bs);
    let mut bc = std::mem::take(&mut h.bc);
    mid.clear();
    mid.resize(bs.len() * mo * mo, Pt::default());
    sfar.clear();
    sfar.resize(bs.len() * fo * fo, Pt::default());
    for ii in 0..h.inst.len() {
        let (nu, nv) = {
            let d = &h.disc[h.inst[ii].patch as usize];
            (d.nu, d.nv)
        };
        for iv in 0..nv {
            for iu in 0..nu {
                let bk = block_of(h, ii as i32, iu, iv) as usize;
                let u0 = -1.0 + 2.0 * iu as f64 / nu as f64;
                let v0 = -1.0 + 2.0 * iv as f64 / nv as f64;
                bs[bk] = rect_size(h, ii as i32, u0, u0 + 2.0 / nu as f64, v0, v0 + 2.0 / nv as f64, &mut bc[bk]);
                let du = 1.0 / nu as f64;
                let dv = 1.0 / nv as f64;
                for a in 0..mo {
                    for c in 0..mo {
                        fill_point(
                            h,
                            ii as i32,
                            iu,
                            iv,
                            u0 + du * (gx[a] + 1.0),
                            v0 + dv * (gx[c] + 1.0),
                            gw[a] * gw[c] * du * dv,
                            true,
                            &mut mid[(bk * mo + a) * mo + c],
                        );
                    }
                }
                for a in 0..fo {
                    for c in 0..fo {
                        fill_point(
                            h,
                            ii as i32,
                            iu,
                            iv,
                            u0 + du * (fx[a] + 1.0),
                            v0 + dv * (fx[c] + 1.0),
                            fw[a] * fw[c] * du * dv,
                            true,
                            &mut sfar[(bk * fo + a) * fo + c],
                        );
                    }
                }
            }
        }
    }
    h.mid = mid;
    h.sfar = sfar;
    h.bs = bs;
    h.bc = bc;
}

/// The compact point arrays of the wave rows (inner and far rule).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn build_wave_arrays(h: &mut HoGeom) -> Result<()> {
    let NQ = h.inner.len();
    let NM = h.NM as usize;
    h.wx.resize(3 * NQ, 0.0);
    h.wn.resize(3 * NQ, 0.0);
    h.wm = vec![0.0; NM * NQ];
    let nb = h.inner0.len() - 1;
    h.bnk = vec![0; nb];
    h.bcol0 = vec![0; nb];
    h.bwb0 = vec![0; nb];
    h.bcol.clear();
    h.wb.clear();
    for qi in 0..NQ {
        let q = &h.inner[qi];
        for k in 0..3 {
            h.wx[3 * qi + k] = q.x[k];
            h.wn[3 * qi + k] = q.n[k];
        }
        for m in 0..NM {
            h.wm[qi * NM + m] = q.w * q.nm[m];
        }
    }
    // far points: Gauss order iqui - 1 (at least 2) on each (instance, panel) block
    {
        // built aside (fill_point reads the geometry), then stored
        let mut fx = Vec::new();
        let mut fn_ = Vec::new();
        let mut fm = Vec::new();
        let mut fwb = Vec::new();
        let mut f0 = vec![0i32; nb + 1];
        for ii in 0..h.inst.len() {
            let d = &h.disc[h.inst[ii].patch as usize];
            let nf = 2.max(d.iqui - 1);
            let (gx, gw) = gauss_legendre(nf);
            for iv in 0..d.nv {
                for iu in 0..d.nu {
                    let bk = block_of(h, ii as i32, iu, iv) as usize;
                    f0[bk] = (fx.len() / 3) as i32;
                    let du = 1.0 / d.nu as f64;
                    let dv = 1.0 / d.nv as f64;
                    for a in 0..nf as usize {
                        for c in 0..nf as usize {
                            let u = -1.0 + du * ((2 * iu + 1) as f64 + gx[a]);
                            let v = -1.0 + dv * ((2 * iv + 1) as f64 + gx[c]);
                            let mut pt = Pt::default();
                            fill_point(h, ii as i32, iu, iv, u, v, gw[a] * gw[c] * du * dv, true, &mut pt);
                            for k in 0..3 {
                                fx.push(pt.x[k]);
                                fn_.push(pt.n[k]);
                            }
                            for m in 0..NM {
                                fm.push(pt.w * pt.nm[m]);
                            }
                            for jb in 0..d.kv as usize {
                                for ja in 0..d.ku as usize {
                                    fwb.push(pt.w * pt.bu[ja] * pt.bv[jb]);
                                }
                            }
                        }
                    }
                }
            }
        }
        f0[nb] = (fx.len() / 3) as i32;
        h.fx = fx;
        h.fn_ = fn_;
        h.fm = fm;
        h.fwb = fwb;
        h.f0 = f0;
    }
    for bk in 0..nb {
        let a = h.inner0[bk] as usize;
        let e = h.inner0[bk + 1] as usize;
        if a == e {
            continue;
        }
        let q0 = &h.inner[a];
        let nk = q0.nk as usize;
        h.bnk[bk] = q0.nk;
        h.bcol0[bk] = h.bcol.len() as i32;
        h.bwb0[bk] = h.wb.len() as i32;
        h.bcol.extend_from_slice(&q0.col[..nk]);
        for qi in a..e {
            let q = &h.inner[qi];
            if q.nk != q0.nk || q.col[..nk] != q0.col[..nk] {
                return Err(Error::internal("inner points of a panel with different basis columns"));
            }
            h.wb.extend_from_slice(&q.wb[..nk]);
        }
    }
    Ok(())
}

/// The parametric mass matrices of the L2 projection (with the area weights), factored once: they
/// depend on the geometry alone. One per patch, from its first instance: a reflection keeps the
/// area weights, so the images have the same matrix. Built only when needed: the scattered
/// diffraction projects the incident wave.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn build_projections(h: &mut HoGeom, needed: bool) -> Result<()> {
    h.proj = vec![RealLu::default(); if needed { h.disc.len() } else { 0 }];
    let mut done = vec![false; h.proj.len()];
    let mut ii = 0;
    while needed && ii < h.inst.len() {
        let p = h.inst[ii].patch as usize;
        if done[p] {
            ii += 1;
            continue;
        }
        done[p] = true;
        let d = &h.disc[p];
        let nb = d.nb() as usize;
        let mu = d.mu();
        let mut Mm = vec![0.0f64; nb * nb];
        let first = block_of(h, ii as i32, 0, 0) as usize;
        let last = first + (d.nu * d.nv) as usize;
        for bk in first..last {
            for qi in h.inner0[bk] as usize..h.inner0[bk + 1] as usize {
                let q = &h.inner[qi];
                for b in 0..d.kv as usize {
                    for a in 0..d.ku as usize {
                        let r = ((q.iv + b as i32) * mu + q.iu + a as i32) as usize;
                        let wr = q.w * q.bu[a] * q.bv[b];
                        for b2 in 0..d.kv as usize {
                            for a2 in 0..d.ku as usize {
                                let c = ((q.iv + b2 as i32) * mu + q.iu + a2 as i32) as usize;
                                Mm[r + c * nb] += wr * q.bu[a2] * q.bv[b2];
                            }
                        }
                    }
                }
            }
        }
        let mut lu = RealLu::default();
        if let Err(e) = lu.factor(nb as i32, Mm) {
            return Err(Error::new(e.kind(), format!("projection on a patch: {}", e.message())));
        }
        h.proj[p] = lu;
        ii += 1;
    }
    Ok(())
}

/// The bytes of those factors (the memory estimate).
pub fn projection_bytes(h: &HoGeom) -> i64 {
    let mut b: i64 = 0;
    for d in &h.disc {
        b += d.nb() as i64 * (d.nb() * 8 + 4) as i64;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_knots_uniform() {
        assert_eq!(open_knots(1, 2), vec![-1.0, -1.0, 1.0, 1.0]);
        assert_eq!(open_knots(4, 3), vec![-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0]);
        // the interior knots computed as -1 + (2 i) / n (the rounding of open_knots)
        let t = open_knots(3, 1);
        assert_eq!(t, vec![-1.0, -1.0 + 2.0 / 3.0, -1.0 + 4.0 / 3.0, 1.0]);
    }

    #[test]
    fn flat_patch_geometry() {
        // a unit square in z = -1 (the GDF vertex order): x(u, v), the tangents and the normal
        let m = Mesh {
            patch: true,
            geometry: PatchKind::Flat,
            count: 1,
            vertices: vec![0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0],
            ..Mesh::default()
        };
        let g = patch_geo(&m, 0);
        assert!(g.flat && !g.analytic && g.mesh.is_none());
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        g.eval(0.0, 0.0, &mut x, &mut xu, &mut xv);
        assert_eq!(x, v3(0.5, 0.5, -1.0));
        assert_eq!(xu, v3(0.5, 0.0, 0.0));
        assert_eq!(xv, v3(0.0, 0.5, 0.0));
        assert_eq!(g.nsign(), 1.0);
    }
}
