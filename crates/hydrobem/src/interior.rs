//! Artificial interior free surface of a body (reference/inputs-bodies §2.6, theory §5.1): the panels of
//! element ranges of the body mesh, of a mesh of its own, or generated from the
//! wetted surface (projected, waterline, graded), in body-local coordinates and in the
//! stored symmetry form of the mesh flags isx, isy (expanded by reflections()
//! like the body panels). Why the extended equation removes the irregular frequencies and why
//! non-conforming triangles of the waterline segments' size suffice: theory §5.2; the grounds of
//! the generated layouts (the refinement factor 2, the sine clustering of `graded`, the cap of
//! 200 000 panels): theory §14.

use crate::model_codes::InteriorMethod;

use crate::constants::PI;
use crate::error::{Error, Result, ResultExt, at_body, at_element};
use crate::geometry::{PLANE_TOL, Panel, Vec3, make_panel, v3};
use crate::mesh::Mesh;
use crate::polygon::{self, P2, Region, inside, orient, signed_area};
use crate::problem::{Problem, Role};

#[derive(Clone, Debug, Default)]
pub struct InteriorPanels {
    pub isx: i32, // symmetry flags of the stored form
    pub isy: i32,
    pub panels: Vec<Panel>, // at z = 0 (body-local), upward normals
}

const SLOPE: f64 = 1e-6; // sine below which a normal counts as horizontal / vertical

fn p2(x: f64, y: f64) -> P2 {
    [x, y]
}
fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn wet_records(m: &Problem, bi: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let role = &m.derived()[bi as usize].role;
    for r in 0..role.len() {
        if role[r] == Role::Wet {
            out.push(r as i32);
        }
    }
    out
}

fn extent(m: &Mesh, recs: &[i32]) -> f64 {
    let mut e = m.reference_length;
    for &r in recs {
        for v in &m.panels[r as usize].raw {
            e = e.max(v[0].abs()).max(v[1].abs()).max(v[2].abs());
        }
    }
    e
}

fn panel_from(v: &[Vec3; 4], at: &str) -> Result<Panel> {
    let mut v12 = [0.0f64; 12];
    for i in 0..4 {
        for k in 0..3 {
            v12[3 * i + k] = v[i][k];
        }
    }
    let mut p = Panel::default();
    let why = make_panel(&v12, &mut p);
    if !why.is_empty() {
        return Err(Error::value(format!("{at}: a generated interior panel is invalid ({why})")));
    }
    Ok(p)
}

// ------------------------------------------------------------------ projected

// Vertical projection of the downward-facing wetted elements onto z = 0
// (normals into the fluid point down); vertical elements are skipped; an
// upward-facing element means that a vertical line meets the body more than
// once and the projection would be multi-valued.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn projected(m: &Mesh, wet: &[i32], z0: f64, tolz: f64, at: &str) -> Result<Vec<Panel>> {
    let mut out = Vec::new();
    for &r in wet {
        let p = &m.panels[r as usize];
        let mut flat = true;
        for v in &p.raw {
            flat = flat && (v[2] - z0).abs() <= tolz;
        }
        if flat {
            continue; // a zero-draft part of the body surface
        }
        if p.normal[2] > SLOPE {
            return Err(Error::value(format!(
                    "{at}: interior surface 'projected': wetted element {} faces upward, so the vertical projection of the wetted surface is multi-valued",
                    r + 1
                )).at(at_element(r + 1)));
        }
        if p.normal[2] >= -SLOPE {
            continue; // vertical
        }
        // reversed order: the projected polygon gets the upward normal
        let mut v = [Vec3::ZERO; 4];
        for i in 0..4 {
            v[i] = v3(p.raw[3 - i][0], p.raw[3 - i][1], z0);
        }
        out.push(panel_from(&v, at)?);
    }
    if out.is_empty() {
        return Err(Error::value(format!("{at}: interior surface 'projected': no wetted element faces downward")));
    }
    Ok(out)
}

// ------------------------------------------------------------------ waterline

type Loop = Vec<P2>;

#[derive(Clone, Copy)]
struct Seg {
    a: P2,
    b: P2,
}

// The waterplane regions of the stored elements, in the stored quadrant mapped
// to x, y >= 0 (s = (sx, sy) maps back): counter-clockwise outer contours,
// each with its clockwise holes; `target` the mean waterline segment length.
struct Regions {
    outers: Vec<Loop>,
    holes: Vec<Vec<Loop>>,
    sx: f64,
    sy: f64,
    target: f64,
    ax_x: bool, // y = 0 resp. x = 0 is a symmetry plane
    ax_y: bool,
    z0: f64, // the body-local height of the free surface
}

// Closed waterline contours of the stored elements (oriented with the body
// section on the left), closed along the symmetry axes; collinear points are
// merged for the triangulation, kept for the fan.
fn regions(m: &Mesh, wet: &[i32], z0: f64, tol: f64, at: &str, name: &str, merge_collinear: bool) -> Result<Regions> {
    let not_close = || Error::value(format!("{at}: interior surface '{name}': the waterline does not close"));
    let (segs, target, sx, sy) = waterline_segments(m, wet, z0, tol, at, name)?;
    let (chains, mut loops) = chain_segments(&segs, tol, at, name)?;
    let ax_x = m.isy != 0;
    let ax_y = m.isx != 0; // y = 0 resp. x = 0 is a symmetry plane
    loops.extend(close_chains(&chains, ax_x, ax_y, tol).ok_or_else(not_close)?);
    clean_loops(&mut loops, tol, merge_collinear);
    // contours that cross or touch (within the tolerance)
    if polygon::loops_meet(&loops, tol) {
        return Err(Error::value(format!("{at}: interior surface '{name}': waterline contours cross or touch")));
    }
    let (outers, holes) = outer_loops_and_holes(loops, at, name)?;
    Ok(Regions { outers, holes, sx, sy, target, ax_x, ax_y, z0 })
}

// The waterline segments of the stored elements (the panel edges in z = z0 of the non-horizontal
// elements), oriented with the body section on the left and mapped to the stored quadrant x, y >= 0:
// (segments, their mean length, sx, sy) with (sx, sy) the map back.
fn waterline_segments(
    m: &Mesh,
    wet: &[i32],
    z0: f64,
    tol: f64,
    at: &str,
    name: &str,
) -> Result<(Vec<Seg>, f64, f64, f64)> {
    let mut segs: Vec<Seg> = Vec::new();
    let mut total = 0.0;
    for &r in wet {
        let p = &m.panels[r as usize];
        let nh = p.normal[0].hypot(p.normal[1]);
        if nh < SLOPE {
            continue;
        }
        let dir = p2(-p.normal[1], p.normal[0]); // z x n_h: the body section on the left
        for i in 0..4 {
            let u = p.raw[i];
            let v = p.raw[(i + 1) % 4];
            if (u[2] - z0).abs() > tol || (v[2] - z0).abs() > tol {
                continue;
            }
            let mut a = p2(u[0], u[1]);
            let mut b = p2(v[0], v[1]);
            if dist2(a, b) <= tol {
                continue;
            }
            if (b[0] - a[0]) * dir[0] + (b[1] - a[1]) * dir[1] < 0.0 {
                std::mem::swap(&mut a, &mut b);
            }
            segs.push(Seg { a, b });
            total += dist2(a, b);
        }
    }
    if segs.is_empty() {
        return Err(Error::value(format!("{at}: interior surface '{name}': the wetted surface has no waterline")));
    }
    let target = total / segs.len() as f64;
    // stored side of the symmetry planes: map to x, y >= 0
    let (mut mx, mut my) = (0.0, 0.0);
    for s in &segs {
        mx += s.a[0] + s.b[0];
        my += s.a[1] + s.b[1];
    }
    let sx = if m.isx != 0 && mx < 0.0 { -1.0 } else { 1.0 };
    let sy = if m.isy != 0 && my < 0.0 { -1.0 } else { 1.0 };
    for s in segs.iter_mut() {
        s.a = p2(sx * s.a[0], sy * s.a[1]);
        s.b = p2(sx * s.b[0], sy * s.b[1]);
        if sx * sy < 0.0 {
            std::mem::swap(&mut s.a, &mut s.b);
        }
    }
    Ok((segs, target, sx, sy))
}

// The segments joined end to start: (the open chains, from a segment without a predecessor, as
// point lists; the closed loops).
fn chain_segments(segs: &[Seg], tol: f64, at: &str, name: &str) -> Result<(Vec<Loop>, Vec<Loop>)> {
    let n = segs.len();
    let mut next = vec![-1i32; n];
    let mut prev = vec![-1i32; n];
    for i in 0..n {
        for j in 0..n {
            if i == j || dist2(segs[i].b, segs[j].a) > tol {
                continue;
            }
            if next[i] >= 0 || prev[j] >= 0 {
                return Err(Error::value(format!(
                    "{at}: interior surface '{name}': waterline contours touch or cross"
                )));
            }
            next[i] = j as i32;
            prev[j] = i as i32;
        }
    }
    let mut used = vec![false; n];
    let mut chains: Vec<Loop> = Vec::new();
    let mut loops: Vec<Loop> = Vec::new();
    for s in 0..n {
        if prev[s] >= 0 {
            continue;
        }
        let mut c: Loop = vec![segs[s].a];
        let mut k = s as i32;
        while k >= 0 {
            used[k as usize] = true;
            c.push(segs[k as usize].b);
            k = next[k as usize];
        }
        chains.push(c);
    }
    for s in 0..n {
        if used[s] {
            continue;
        }
        let mut c: Loop = Vec::new();
        let mut k = s as i32;
        while !used[k as usize] {
            used[k as usize] = true;
            c.push(segs[k as usize].a);
            k = next[k as usize];
            if k < 0 {
                return Err(Error::internal("interior surface generation: broken waterline loop"));
            }
        }
        loops.push(c);
    }
    Ok((chains, loops))
}

// The loops that close the open chains along the symmetry axes (the boundary of the stored
// quadrant traversed with the region on the left: +x along y = 0, then -y along x = 0; with one
// plane only, the stored half: its whole line, the region on the left); None when they do not
// close. ax_x, ax_y: y = 0 resp. x = 0 is a symmetry plane.
fn close_chains(chains: &[Loop], ax_x: bool, ax_y: bool, tol: f64) -> Option<Vec<Loop>> {
    let on_x = |p: P2| ax_x && p[1].abs() <= tol && (p[0] >= -tol || !ax_y);
    let on_y = |p: P2| ax_y && p[0].abs() <= tol && (p[1] >= -tol || !ax_x);
    let key_start = |p: P2| if on_x(p) { (0, p[0]) } else { (1, -p[1]) };
    let key_end = |p: P2| if on_y(p) { (1, -p[1]) } else { (0, p[0]) };
    for c in chains {
        let (f, b) = (c[0], c[c.len() - 1]);
        if !(on_x(f) || on_y(f)) || !(on_x(b) || on_y(b)) {
            return None;
        }
    }
    let mut loops: Vec<Loop> = Vec::new();
    let mut cused = vec![false; chains.len()];
    for c0 in 0..chains.len() {
        if cused[c0] {
            continue;
        }
        let mut l: Loop = Vec::new();
        let mut c = c0;
        loop {
            cused[c] = true;
            l.extend_from_slice(&chains[c]);
            let ke = key_end(chains[c][chains[c].len() - 1]);
            // next chain start along the boundary; past the end of the boundary
            // it wraps through the corner of the quadrant
            let mut best: i32 = -1;
            let mut wrap = false;
            let mut pass = 0;
            while pass < 2 && best < 0 {
                for d in 0..chains.len() {
                    if cused[d] && d != c0 {
                        continue;
                    }
                    let ks = key_start(chains[d][0]);
                    if pass == 0 && ks < ke {
                        continue;
                    }
                    if best < 0 || ks < key_start(chains[best as usize][0]) {
                        best = d as i32;
                    }
                }
                wrap = pass == 1;
                pass += 1;
            }
            if best < 0 {
                return None;
            }
            let ks = key_start(chains[best as usize][0]);
            if ke.0 == 0 && ks.0 == 1 && !wrap {
                return None;
            }
            if wrap {
                if !(ax_x && ax_y) {
                    return None;
                }
                l.push(p2(0.0, 0.0)); // the corner of the quadrant
            }
            if best as usize == c0 {
                break;
            }
            c = best as usize;
        }
        loops.push(l);
    }
    Some(loops)
}

// Drops repeated points (and the closing repetition) of every loop and, with merge_collinear,
// its collinear points.
fn clean_loops(loops: &mut [Loop], tol: f64, merge_collinear: bool) {
    for l in loops.iter_mut() {
        let mut c: Loop = Vec::new();
        for &p in l.iter() {
            if c.is_empty() || dist2(c[c.len() - 1], p) > tol {
                c.push(p);
            }
        }
        while c.len() > 1 && dist2(c[0], c[c.len() - 1]) <= tol {
            c.pop();
        }
        *l = if merge_collinear { polygon::merge_collinear(&c, tol) } else { c };
    }
}

// The outer loops (counter-clockwise) and the holes (clockwise) of each, a hole belonging to the
// smallest outer loop around it.
fn outer_loops_and_holes(loops: Vec<Loop>, at: &str, name: &str) -> Result<(Vec<Loop>, Vec<Vec<Loop>>)> {
    let mut outers: Vec<Loop> = Vec::new();
    let mut holes: Vec<Loop> = Vec::new();
    for l in loops {
        if l.len() < 3 {
            continue;
        }
        if signed_area(&l) > 0.0 { outers.push(l) } else { holes.push(l) }
    }
    if outers.is_empty() {
        return Err(Error::value(format!("{at}: interior surface '{name}': no enclosed region")));
    }
    let mut hole_of: Vec<Vec<Loop>> = vec![Vec::new(); outers.len()];
    for h in holes {
        let mut owner: i32 = -1;
        for o in 0..outers.len() {
            if inside(&outers[o], h[0])
                && (owner < 0 || signed_area(&outers[o]).abs() < signed_area(&outers[owner as usize]).abs())
            {
                owner = o as i32;
            }
        }
        if owner < 0 {
            return Err(Error::value(format!(
                "{at}: interior surface '{name}': a waterline contour lies outside the body"
            )));
        }
        hole_of[owner as usize].push(h);
    }
    Ok((outers, hole_of))
}

// A generated polygon (3 or 4 corners, counter-clockwise in the mapped
// quadrant) as a panel of the stored quadrant.
fn mapped_panel(g: &Regions, c: &[P2], at: &str) -> Result<Panel> {
    let mut v = [Vec3::ZERO; 4];
    for k in 0..c.len() {
        v[k] = v3(g.sx * c[k][0], g.sy * c[k][1], g.z0);
    }
    if g.sx * g.sy < 0.0 {
        v[..c.len()].reverse();
    }
    if c.len() == 3 {
        v[3] = v[2];
    }
    panel_from(&v, at)
}

// Waterline generator: the regions triangulated with holes and refined
// (polygon::triangulate_refined: bridges, ear clipping, Delaunay flips, then Delaunay refinement to
// edges of at most twice the mean waterline segment and the smallest angle REFINE_MIN_ANGLE).
fn waterline(m: &Mesh, wet: &[i32], z0: f64, tol: f64, at: &str) -> Result<Vec<Panel>> {
    let g = regions(m, wet, z0, tol, at, "waterline", true)?;
    // more panels than MAX_GENERATED would make the dense system of the solve impractical (its
    // matrix alone 200000^2 complex entries), so a region that needs them is refused, not left
    // coarser
    const MAX_GENERATED: usize = 200000;
    let mut done: Vec<[P2; 3]> = Vec::new();
    for o in 0..g.outers.len() {
        // the vertices of the cleaned loops lie more than tol apart: within it they are one (the
        // two ends of a bridge)
        let r = Region { outer: g.outers[o].clone(), holes: g.holes[o].clone() };
        done.extend(polygon::triangulate_refined(&r, tol, 2.0 * g.target, MAX_GENERATED - done.len())?);
        if done.len() > MAX_GENERATED {
            return Err(Error::value(format!(
                "{at}: interior surface 'waterline': the waterplane needs more than {MAX_GENERATED} panels of twice \
                 the mean waterline segment length; use 'graded' or a mesh of its own"
            )));
        }
    }
    let mut out = Vec::new();
    for t in &done {
        out.push(mapped_panel(&g, &[t[0], t[1], t[2]], at)?);
    }
    Ok(out)
}

// ------------------------------------------------------------------ graded

// Graded generator: each region without openings is fanned from its
// area centroid (on a symmetry axis the region touches: the centroid of the
// reflected region) to every waterline segment; the rays are divided at
// t_k = sin(pi k / (2 N)), k = 0..N, dense towards the waterline, with one N
// per region, N = ceil(mean ray length / mean segment length). The innermost
// cell of a segment is a triangle, the others are quadrilaterals; together they
// cover the polygonal waterplane exactly.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn graded(m: &Mesh, wet: &[i32], z0: f64, tol: f64, at: &str) -> Result<Vec<Panel>> {
    let g = regions(m, wet, z0, tol, at, "graded", false)?;
    let mut out = Vec::new();
    for o in 0..g.outers.len() {
        let l = &g.outers[o];
        if !g.holes[o].is_empty() {
            return Err(Error::value(format!(
                "{at}: interior surface 'graded': a waterplane region has an opening (a moonpool); use 'waterline'"
            )));
        }
        // area centroid, moved onto the symmetry axes the region lies on
        let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
        let (mut on_y0, mut on_x0) = (false, false);
        for i in 0..l.len() {
            let p = l[i];
            let q = l[(i + 1) % l.len()];
            let c = p[0] * q[1] - p[1] * q[0];
            a += 0.5 * c;
            cx += (p[0] + q[0]) * c / 6.0;
            cy += (p[1] + q[1]) * c / 6.0;
            on_y0 = on_y0 || (g.ax_x && p[1].abs() <= tol && q[1].abs() <= tol);
            on_x0 = on_x0 || (g.ax_y && p[0].abs() <= tol && q[0].abs() <= tol);
        }
        let mut c = p2(cx / a, cy / a);
        if on_x0 {
            c[0] = 0.0;
        }
        if on_y0 {
            c[1] = 0.0;
        }
        // the fanned edges: the waterline segments; edges through the centroid
        // (the closing edges on the symmetry axes) bound no area
        let mut fan: Vec<(P2, P2)> = Vec::new();
        let (mut ray, mut seg) = (0.0, 0.0);
        for i in 0..l.len() {
            let p = l[i];
            let q = l[(i + 1) % l.len()];
            let s = orient(c, p, q);
            if s.abs() <= tol * dist2(p, q) {
                continue;
            }
            if s < 0.0 {
                return Err(Error::value(format!(
                    "{at}: interior surface 'graded': a ray from the centroid of a waterplane region meets its waterline more than once; use 'waterline'"
                )));
            }
            fan.push((p, q));
            ray += 0.5 * (dist2(c, p) + dist2(c, q));
            seg += dist2(p, q);
        }
        if fan.is_empty() {
            return Err(Error::value(format!("{at}: interior surface 'graded': no enclosed region")));
        }
        let N = 1.max((ray / seg - 1e-9).ceil() as i32);
        let mut t = vec![0.0; (N + 1) as usize];
        for k in 0..=N {
            t[k as usize] = if k == N { 1.0 } else { (0.5 * PI * k as f64 / N as f64).sin() };
        }
        let at_t = |p: P2, s: f64| p2(c[0] + s * (p[0] - c[0]), c[1] + s * (p[1] - c[1]));
        for &(p, q) in &fan {
            out.push(mapped_panel(&g, &[c, at_t(p, t[1]), at_t(q, t[1])], at)?);
            for k in 1..N as usize {
                out.push(mapped_panel(&g, &[at_t(p, t[k]), at_t(p, t[k + 1]), at_t(q, t[k + 1]), at_t(q, t[k])], at)?);
            }
        }
    }
    Ok(out)
}

/// The interior free surface of a body, selected, external or generated.
/// m: problem; body: zero-based body index. Returns the panels in body-local coordinates,
/// stored symmetry form. Fails with ErrorKind::Value for geometry a generator cannot handle (multi-valued
/// projection, open or crossing waterlines, regions the graded fan cannot cover),
/// ErrorKind::Unsupported for patch meshes.
pub fn interior_panels(m: &Problem, body: i32) -> Result<InteriorPanels> {
    let bi = body;
    let here = at_body(bi);
    let b = &m.bodies[bi as usize];
    let bm = &m.meshes[b.mesh as usize];
    let mut out = InteriorPanels::default();
    let at = format!("body '{}'", b.name);
    let low_order = |mm: &Mesh| {
        if mm.patch {
            return Err(Error::unsupported(format!("{at}: generated interior surfaces need a low-order mesh")).at(here));
        }
        Ok(())
    };
    match b.interior_method {
        InteriorMethod::None => {}
        InteriorMethod::PanelRange => {
            low_order(bm)?;
            out.isx = bm.isx;
            out.isy = bm.isy;
            for r in &b.interior_panels {
                for rec in r.first - 1..r.last {
                    out.panels.push(bm.panels[rec as usize]);
                }
            }
        }
        InteriorMethod::ExternalMesh => {
            let im = &m.meshes[b.interior_mesh.expect("an external interior surface has its mesh (rule_bodies)")];
            low_order(im)?;
            out.isx = im.isx;
            out.isy = im.isy;
            out.panels = im.panels.clone();
        }
        InteriorMethod::Projected | InteriorMethod::Waterline | InteriorMethod::Graded => {
            low_order(bm)?;
            out.isx = bm.isx;
            out.isy = bm.isy;
            let wet = wet_records(m, bi);
            let tol = PLANE_TOL * extent(bm, &wet);
            // the free surface in body coordinates: z = -position z (the body placed there)
            let z0 = -b.position[2];
            out.panels = match b.interior_method {
                InteriorMethod::Projected => projected(bm, &wet, z0, tol, &at),
                InteriorMethod::Waterline => waterline(bm, &wet, z0, tol, &at),
                _ => graded(bm, &wet, z0, tol, &at),
            }
            .located(here)?;
        }
    }
    Ok(out)
}
