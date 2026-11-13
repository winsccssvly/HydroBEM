//! The rules of the free-surface plates in the plane z = 0 (reference/inputs-rules §1, theory §9.7): a plate
//! overlaps no other surface of z = 0 (another plate, a lid, an interior surface, a pressure
//! surface in z = 0), and a thin sheet meets a plate or a lid only along the edges of its elements.
//!
//! A point of z = 0 has one boundary condition: a plate over a lid or an interior surface poses
//! two conditions on the same water and its equations are not those of either surface. Overlaps
//! are measured by the convex clipping of the elements (Sutherland and Hodgman 1974) as horizontal
//! polygons: an overlap whose width (its area over its diameter) exceeds a tenth of the smaller
//! element's diameter is refused; edges touching, and the slivers between two polygons that
//! approximate one curve by different chords, pass.
//!
//! The potential of a plate or a lid jumps across a sheet that hangs from it (the water on the two
//! sides of the sheet is not the same): an element of constant potential (low order) or of a smooth
//! potential (higher order) cannot hold the jump inside it, and the solution then does not converge
//! as the mesh is refined (theory §9.6). A sheet's edge in z = 0 must therefore lie on the edges of
//! the elements of the plate or lid it meets.

use crate::error::{Error, Result};
use crate::geometry::{Placement, Reflection, Vec3, reflect_vertices, reflections, v3};
use crate::interior::interior_panels;
use crate::mesh::{Mesh, patch_eval};
use crate::mesh_desc::PatchKind;
use crate::model_codes::{Domain, InteriorMethod};
use crate::plate_geometry::{Rec, Winding, ranges_records, rec_of, record};
use crate::polygon::{BoxTree, P2, bounds, inside, signed_area};
use crate::problem::{Problem, Role, body_label, model_place, plane_tol};
use crate::text::element_text;

/// The width of an overlap, relative to the smaller element's diameter, above which it is refused.
const OVERLAP_WIDTH: f64 = 0.1;
/// The distance from an element's edges, relative to its diameter, within which a sheet's edge counts
/// as lying on them.
const EDGE_TOL: f64 = 1e-2;

/// A surface of z = 0 and its elements, and the body it moves with (-1: a lid not placed with a
/// body).
struct Cover {
    label: String,
    body: i32,
    plate: bool,
    sheet_column: bool, // a plate or a lid: a sheet meeting it must end on its edges
    recs: Vec<Rec>,
}

fn records(p: &Problem, mi: usize, recs: &[i32], place: &Placement) -> Vec<Rec> {
    let m = &p.meshes[mi];
    let mut out = Vec::new();
    for r in reflections(m.isx, m.isy) {
        for &rec in recs {
            out.push(record(m, rec as usize, &r, place, p.stored_record(mi as i32, rec)));
        }
    }
    out
}

fn in_z0(m: &Mesh, rec: usize, z: f64) -> bool {
    let sh = &m.shapes[rec];
    !sh.hull.is_empty() && sh.hull.iter().all(|v| (v[2] + z).abs() <= plane_tol(m, sh))
}

/// The surfaces of z = 0 of the problem.
fn covers(p: &Problem) -> Vec<Cover> {
    let mut out = Vec::new();
    for (bi, b) in p.bodies.iter().enumerate() {
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        let at = body_label(b);
        for (s, pl) in b.plates.iter().enumerate() {
            let (mi, recs) = if let Some(pi) = pl.mesh {
                (pi, (0..p.meshes[pi].count).collect())
            } else {
                (b.mesh as usize, ranges_records(&pl.panels))
            };
            out.push(Cover {
                label: format!("{at} {}", element_text("free-surface plate", model_place(pl.place, s), &pl.name)),
                body: bi as i32,
                plate: true,
                sheet_column: true,
                recs: records(p, mi, &recs, &place),
            });
        }
        for (s, ps) in b.pressure.iter().enumerate() {
            let mi = ps.mesh.unwrap_or(b.mesh as usize);
            let all: Vec<i32> =
                if ps.mesh.is_some() { (0..p.meshes[mi].count).collect() } else { ranges_records(&ps.panels) };
            let recs: Vec<i32> = all.into_iter().filter(|&r| in_z0(&p.meshes[mi], r as usize, b.position[2])).collect();
            if !recs.is_empty() {
                out.push(Cover {
                    label: format!("{at} {}", element_text("pressure surface", model_place(ps.place, s), &ps.name)),
                    body: bi as i32,
                    plate: false,
                    sheet_column: false,
                    recs: records(p, mi, &recs, &place),
                });
            }
        }
        let interior = format!("{at} interior surface");
        match b.interior_method {
            InteriorMethod::None => {}
            InteriorMethod::PanelRange => out.push(Cover {
                label: interior,
                body: bi as i32,
                plate: false,
                sheet_column: false,
                recs: records(p, b.mesh as usize, &ranges_records(&b.interior_panels), &place),
            }),
            InteriorMethod::ExternalMesh => {
                let mi = b.interior_mesh.expect("an external interior surface has its mesh (rule_bodies)");
                out.push(Cover {
                    label: interior,
                    body: bi as i32,
                    plate: false,
                    sheet_column: false,
                    recs: records(p, mi, &(0..p.meshes[mi].count).collect::<Vec<_>>(), &place),
                });
            }
            _ => {
                // a generated interior surface (its own rules report a waterline it cannot fill)
                if let Ok(ip) = interior_panels(p, bi as i32) {
                    let mut recs = Vec::new();
                    for r in reflections(ip.isx, ip.isy) {
                        for (k, pn) in ip.panels.iter().enumerate() {
                            let v: Vec<P2> = reflect_vertices(&pn.raw, &r)
                                .iter()
                                .map(|x| {
                                    let y = place.apply(*x);
                                    [y[0], y[1]]
                                })
                                .collect();
                            recs.push(rec_of(vec![v.clone()], v, k as i32 + 1));
                        }
                    }
                    out.push(Cover { label: interior, body: bi as i32, plate: false, sheet_column: false, recs });
                }
            }
        }
    }
    for l in &p.lids {
        let (mi, recs, place) = if l.body >= 0 {
            let b = &p.bodies[l.body as usize];
            (b.mesh as usize, ranges_records(&l.panels), Placement { position: b.position, yaw_deg: b.yaw })
        } else {
            let lm = l.mesh.expect("a lid of its own has its mesh (rule_lids)");
            (lm, (0..p.meshes[lm].count).collect(), Placement::default())
        };
        out.push(Cover {
            label: format!("lids/{}", l.name),
            body: l.body,
            plate: false,
            sheet_column: true,
            recs: records(p, mi, &recs, &place),
        });
    }
    out
}

/// The part of the convex polygon `a` inside the convex polygon `c` (both counter-clockwise).
fn clip(a: &[P2], c: &[P2]) -> Vec<P2> {
    let mut out = a.to_vec();
    for k in 0..c.len() {
        if out.is_empty() {
            break;
        }
        let (p, q) = (c[k], c[(k + 1) % c.len()]);
        let side = |x: P2| (q[0] - p[0]) * (x[1] - p[1]) - (q[1] - p[1]) * (x[0] - p[0]);
        let input = std::mem::take(&mut out);
        for i in 0..input.len() {
            let (s, e) = (input[i], input[(i + 1) % input.len()]);
            let (ds, de) = (side(s), side(e));
            if ds >= 0.0 {
                out.push(s);
            }
            if (ds >= 0.0) != (de >= 0.0) {
                let t = ds / (ds - de);
                out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
            }
        }
    }
    out
}

fn diameter(l: &[P2]) -> f64 {
    let mut d: f64 = 0.0;
    for a in l {
        for b in l {
            d = d.max((a[0] - b[0]).hypot(a[1] - b[1]));
        }
    }
    d
}

/// The width of the overlap of two elements (the widest of their pieces' overlaps), relative to the
/// smaller element's diameter.
fn overlap(a: &Rec, b: &Rec) -> f64 {
    let mut w: f64 = 0.0;
    for pa in &a.pieces {
        for pb in &b.pieces {
            let x = clip(pa, pb);
            if x.len() >= 3 {
                let d = diameter(&x);
                if d > 0.0 {
                    w = w.max(signed_area(&x).abs() / d);
                }
            }
        }
    }
    w / a.size.min(b.size)
}

fn boxes_meet(a: &Rec, b: &Rec) -> bool {
    a.lo[0] < b.hi[0] && b.lo[0] < a.hi[0] && a.lo[1] < b.hi[1] && b.lo[1] < a.hi[1]
}

/// The distance from x to the segment ab, as the rules of the plates have always computed it:
/// x - a - t e. `polygon::segment_distance` (x - (a + t e)) can differ in the last bit, and the
/// rules' tolerances (EDGE_TOL, the edge test, WALL_TOL) compare against this exact rounding, so
/// that a refusal at a tolerance does not flip; the two are kept apart.
fn segment_distance(x: P2, a: P2, b: P2) -> f64 {
    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
    let l2 = ex * ex + ey * ey;
    let t = if l2 > 0.0 { (((x[0] - a[0]) * ex + (x[1] - a[1]) * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (x[0] - a[0] - t * ex).hypot(x[1] - a[1] - t * ey)
}

/// Whether x lies inside the element, away from its edges.
fn strictly_inside(r: &Rec, x: P2) -> bool {
    if x[0] < r.lo[0] || x[0] > r.hi[0] || x[1] < r.lo[1] || x[1] > r.hi[1] {
        return false;
    }
    if !r.pieces.iter().any(|p| inside(p, x)) {
        return false;
    }
    let n = r.boundary.len();
    (0..n).all(|k| segment_distance(x, r.boundary[k], r.boundary[(k + 1) % n]) > EDGE_TOL * r.size)
}

/// The edges in z = 0 of element rec of mesh m reflected by r and placed: their end points (a
/// curved patch's sides in 32 chords each).
fn edges_in_z0(m: &Mesh, rec: usize, r: &Reflection, place: &Placement) -> Vec<(P2, P2)> {
    let sh = &m.shapes[rec];
    let tol = plane_tol(m, sh);
    let pts: Vec<Vec3> = if !m.patch || (m.geometry == PatchKind::Flat && !m.trimmed()) {
        let v: [Vec3; 4] = if m.patch {
            std::array::from_fn(|k| {
                v3(m.vertices[12 * rec + 3 * k], m.vertices[12 * rec + 3 * k + 1], m.vertices[12 * rec + 3 * k + 2])
            })
        } else {
            [sh.hull[0], sh.hull[1], sh.hull[2], sh.hull[3]]
        };
        reflect_vertices(&v, r).iter().map(|x| place.apply(*x)).collect()
    } else {
        let n = 32;
        let at = |u: f64, w: f64| {
            let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            patch_eval(m, rec as i32, u, w, &mut x, &mut xu, &mut xv);
            place.apply(v3(r.sx * x[0], r.sy * x[1], x[2]))
        };
        let s = |k: usize| -1.0 + 2.0 * k as f64 / n as f64;
        let mut b: Vec<Vec3> = (0..n).map(|k| at(s(k), -1.0)).collect();
        b.extend((0..n).map(|k| at(1.0, s(k))));
        b.extend((0..n).map(|k| at(-s(k), 1.0)));
        b.extend((0..n).map(|k| at(-1.0, -s(k))));
        b
    };
    let mut out = Vec::new();
    for k in 0..pts.len() {
        let (a, c) = (pts[k], pts[(k + 1) % pts.len()]);
        if a[2].abs() <= tol && c[2].abs() <= tol && (a[0] - c[0]).hypot(a[1] - c[1]) > 0.0 {
            out.push(([a[0], a[1]], [c[0], c[1]]));
        }
    }
    out
}

/// The edges in z = 0 of the thin sheets: (the sheet in the messages, its edge's end points).
fn sheet_edges(p: &Problem) -> Vec<(String, P2, P2)> {
    let mut out = Vec::new();
    let mut add = |m: &Mesh, rec: usize, r: &Reflection, place: &Placement, what: String| {
        for (a, c) in edges_in_z0(m, rec, r, place) {
            out.push((what.clone(), a, c));
        }
    };
    let derived = p.derived();
    for (bi, b) in p.bodies.iter().enumerate() {
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        let m = &p.meshes[b.mesh as usize];
        let d = &derived[bi];
        for r in reflections(m.isx, m.isy) {
            for rec in 0..m.count as usize {
                if d.role[rec] == Role::Dipole && d.tank[rec] < 0 {
                    let what = format!("{}: element {}", body_label(b), p.stored_record(b.mesh, rec as i32));
                    add(m, rec, &r, &place, what);
                }
            }
        }
        if let Some(dm) = b.dipole_mesh {
            let dm = &p.meshes[dm];
            for r in reflections(dm.isx, dm.isy) {
                for rec in 0..dm.count as usize {
                    add(
                        dm,
                        rec,
                        &r,
                        &place,
                        format!("{}: element {} of the mesh of its dipoles", body_label(b), rec + 1),
                    );
                }
            }
        }
    }
    for (k, w) in p.fixed_dampers.iter().enumerate() {
        let dm = &p.meshes[w.mesh as usize];
        for r in reflections(dm.isx, dm.isy) {
            for rec in 0..dm.count as usize {
                let what = format!("fixed_dampers {}: element {}", model_place(w.place, k) + 1, rec + 1);
                add(dm, rec, &r, &Placement::default(), what);
            }
        }
    }
    out
}

/// Whether x lies on the element: inside it or within EDGE_TOL of its diameter from its edges.
fn on_element(r: &Rec, x: P2) -> bool {
    let t = EDGE_TOL * r.size;
    if x[0] < r.lo[0] - t || x[0] > r.hi[0] + t || x[1] < r.lo[1] - t || x[1] > r.hi[1] + t {
        return false;
    }
    let n = r.boundary.len();
    r.pieces.iter().any(|p| inside(p, x))
        || (0..n).any(|k| segment_distance(x, r.boundary[k], r.boundary[(k + 1) % n]) <= t)
}

/// A thin sheet hanging from a free-surface plate (its edge in z = 0 on the plate's elements, inside
/// or along their edges): the sheet in the messages, the plate and its body, else None. At omega = inf the
/// plate is a sheet of the odd-image problem, and the junction of the hanging sheet, its image and
/// the plate has no rule (theory §9.7).
pub(crate) fn sheet_on_plate(p: &Problem) -> Option<(String, String, i32)> {
    if !p.bodies.iter().any(|b| !b.plates.is_empty()) {
        return None;
    }
    let edges = sheet_edges(p);
    if edges.is_empty() {
        return None;
    }
    let cv = covers(p);
    let plates: Vec<&Cover> = cv.iter().filter(|c| c.plate).collect();
    let ends: Vec<(P2, P2)> = edges.iter().map(|e| (e.1, e.2)).collect();
    let (ei, ci, _) = sheet_on_plate_indexed(&ends, &plates)?;
    Some((edges[ei].0.clone(), plates[ci].label.clone(), plates[ci].body))
}

/// The first point at a quarter, half or three quarters of a sheet edge (the edges in order, for
/// each the surfaces 0..n in order, then the three points) that `meets(surface, point)` places on
/// an element of a surface: (edge, surface, element).
fn sheet_meeting(
    edges: &[(P2, P2)],
    n: usize,
    meets: &dyn Fn(usize, P2) -> Option<usize>,
) -> Option<(usize, usize, usize)> {
    for (ei, &(a, b)) in edges.iter().enumerate() {
        for c in 0..n {
            for t in [0.25, 0.5, 0.75] {
                let x = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                if let Some(r) = meets(c, x) {
                    return Some((ei, c, r));
                }
            }
        }
    }
    None
}

/// The tree of the boxes of a surface's elements, each grown by `grow` of it (its search selects
/// the candidates of a test that is false outside that grown box).
fn element_tree(c: &Cover, grow: impl Fn(&Rec) -> f64) -> BoxTree {
    let boxes: Vec<(P2, P2)> = c
        .recs
        .iter()
        .map(|r| {
            let t = grow(r);
            ([r.lo[0] - t, r.lo[1] - t], [r.hi[0] + t, r.hi[1] + t])
        })
        .collect();
    BoxTree::new(&boxes)
}

/// sheet_on_plate's search (sheet_meeting, `on_element`) through trees of the plate elements'
/// boxes grown by EDGE_TOL of their diameters: the same first edge and plate as a pass over every
/// element (the element named is any one that holds the point).
fn sheet_on_plate_indexed(edges: &[(P2, P2)], plates: &[&Cover]) -> Option<(usize, usize, usize)> {
    let trees: Vec<BoxTree> = plates.iter().map(|c| element_tree(c, |r| EDGE_TOL * r.size)).collect();
    sheet_meeting(edges, plates.len(), &|c, x| {
        let mut at = None;
        trees[c].any(x, x, |i| {
            let on = on_element(&plates[c].recs[i], x);
            if on {
                at = Some(i);
            }
            on
        });
        at
    })
}

/// check's search of a sheet edge inside an element of a plate or a lid (sheet_meeting,
/// `strictly_inside`, the first such element of the surface) through trees of the elements'
/// boxes: the same edge, surface and element as a pass over every element.
fn sheet_inside_indexed(edges: &[(P2, P2)], columns: &[&Cover]) -> Option<(usize, usize, usize)> {
    let trees: Vec<BoxTree> = columns.iter().map(|c| element_tree(c, |_| 0.0)).collect();
    sheet_meeting(edges, columns.len(), &|c, x| {
        trees[c].search(x, x).into_iter().find(|&i| strictly_inside(&columns[c].recs[i], x))
    })
}

/// The distance from a plate edge's midpoint to a wall, relative to the plate edge's length, within
/// which the edge lies on the wall: the chords of two meshes of one curve of radius R. A chord of
/// length l lies up to l^2/(8 R) inside the curve, a tenth of l for a chord angle l/R up to 0.8
/// (46 degrees, 8 chords a circle). A plate edge whose ends lie on the wall's curve is thus within
/// a tenth of its length of a wall as fine as the plate or finer; the wall's own length is no part
/// of the bound (a long straight wall edge has no sag, and with it the bound would hold a plate
/// beside the wall: a mat floating in a gap, the edges across it).
const WALL_TOL: f64 = 0.1;

/// Whether a plate edge of midpoint mid and length len lies on the wall edge (wa, wb).
fn on_wall_edge(mid: P2, len: f64, wa: P2, wb: P2) -> bool {
    segment_distance(mid, wa, wb) <= WALL_TOL * len
}

/// The walls of the domain a plate edge may lie on: the planes x = 0, y = 0 of the walls, the sides
/// y = ±half of a channel (NaN without).
struct DomainWalls {
    x: bool,
    y: bool,
    half: f64,
}

impl DomainWalls {
    fn of(p: &Problem) -> Self {
        let o = &p.opt;
        DomainWalls {
            x: o.domain == Domain::XWall || o.domain == Domain::OrthogonalWalls,
            y: o.domain == Domain::YWall || o.domain == Domain::OrthogonalWalls,
            half: if o.domain == Domain::Channel { 0.5 * o.channel_width } else { f64::NAN },
        }
    }

    fn holds(&self, a: P2, b: P2, len: f64) -> bool {
        let on_plane = |u: f64, v: f64, at: f64| (u - at).abs() <= 1e-6 * len && (v - at).abs() <= 1e-6 * len;
        (self.x && on_plane(a[0], b[0], 0.0))
            || (self.y && on_plane(a[1], b[1], 0.0))
            || (self.half.is_finite() && (on_plane(a[1], b[1], self.half) || on_plane(a[1], b[1], -self.half)))
    }
}

/// The edges of the plate elements bordered by free surface, (cover, element, edge) in the order of
/// the covers, their elements and the elements' boundaries (only the first when `first`): an edge
/// with the point a tenth of its length outside its midpoint on no plate element of the same body
/// (`shared`, the point and the body), not on a wall of the domain and with no wall at its midpoint
/// (`on_wall`, its midpoint and length). Beside a plate of another body the water's vertical
/// velocity jumps from one plate's motion to the other's, as at an edge bordered by free surface
/// (theory §9.7): such an edge is free.
fn free_edges(
    plates: &[&Cover],
    domain: &DomainWalls,
    first: bool,
    shared: &dyn Fn(P2, i32) -> bool,
    on_wall: &dyn Fn(P2, f64) -> bool,
) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (ci, c) in plates.iter().enumerate() {
        for (ri, r) in c.recs.iter().enumerate() {
            let l = &r.boundary;
            let s = if signed_area(l) >= 0.0 { 1.0 } else { -1.0 };
            for k in 0..l.len() {
                let (a, b) = (l[k], l[(k + 1) % l.len()]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let len = dx.hypot(dy);
                if len <= 1e-9 * r.size {
                    continue;
                }
                let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
                // outward of a counter-clockwise boundary: the edge turned clockwise
                let out_point = [mid[0] + 0.1 * s * dy, mid[1] - 0.1 * s * dx];
                if !shared(out_point, c.body) && !domain.holds(a, b, len) && !on_wall(mid, len) {
                    out.push((ci, ri, k));
                    if first {
                        return out;
                    }
                }
            }
        }
    }
    out
}

/// The free edges (free_edges) found through trees of the plate pieces' boxes and of the wall
/// edges' boxes: the same edges as a pass over every piece and every wall for each edge, in about
/// n log n steps.
fn free_edges_indexed(
    plates: &[&Cover],
    walls: &[(P2, P2)],
    domain: &DomainWalls,
    first: bool,
) -> Vec<(usize, usize, usize)> {
    let pieces: Vec<(&Vec<P2>, i32)> =
        plates.iter().flat_map(|c| c.recs.iter().flat_map(|r| r.pieces.iter().map(|pc| (pc, c.body)))).collect();
    let pieces_tree = BoxTree::new(&pieces.iter().map(|pc| bounds(pc.0)).collect::<Vec<_>>());
    let walls_tree = BoxTree::new(
        &walls
            .iter()
            .map(|&(a, b)| ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])]))
            .collect::<Vec<_>>(),
    );
    // a wall within WALL_TOL of the plate edge's length: its box meets mid's box grown by WALL_TOL len
    let shared = |x: P2, body: i32| pieces_tree.any(x, x, |i| pieces[i].1 == body && inside(pieces[i].0, x));
    let on_wall = |mid: P2, len: f64| {
        let g = WALL_TOL * len;
        walls_tree
            .any([mid[0] - g, mid[1] - g], [mid[0] + g, mid[1] + g], |i| on_wall_edge(mid, len, walls[i].0, walls[i].1))
    };
    free_edges(plates, domain, first, &shared, &on_wall)
}

/// The free edges (free_edges) by a pass over every plate piece and every wall for each edge: the
/// reference of the indexed search in the tests.
#[cfg(test)]
fn free_edges_brute(
    plates: &[&Cover],
    walls: &[(P2, P2)],
    domain: &DomainWalls,
    first: bool,
) -> Vec<(usize, usize, usize)> {
    let shared = |x: P2, body: i32| {
        plates.iter().any(|q| q.body == body && q.recs.iter().any(|t| t.pieces.iter().any(|pc| inside(pc, x))))
    };
    let on_wall = |mid: P2, len: f64| walls.iter().any(|&(wa, wb)| on_wall_edge(mid, len, wa, wb));
    free_edges(plates, domain, first, &shared, &on_wall)
}

/// The first edge of a free-surface plate bordered by free surface (theory §9.7, the edge kinds):
/// an edge of a plate element in z = 0 with no other plate element of its body beside it (the
/// point a tenth of the edge's length outside its midpoint on no plate element of that body: beside
/// another body's plate the vertical velocity jumps as beside free surface) and no wall at its
/// midpoint (a waterline edge of a wetted element of any body within WALL_TOL, or a wall of the
/// domain). The plate in the messages, the element's number and the plate's body; None when every
/// edge lies on a wall or between plate elements of one body (the plate's edge kinds a and b: on a
/// wall of its own body, or one it slides along).
pub(crate) fn free_plate_edge(p: &Problem) -> Option<(String, i32, i32)> {
    if !p.bodies.iter().any(|b| !b.plates.is_empty()) {
        return None;
    }
    let cv = covers(p);
    let plates: Vec<&Cover> = cv.iter().filter(|c| c.plate).collect();
    // the waterline edges of the wetted elements of every body
    let mut walls: Vec<(P2, P2)> = Vec::new();
    let derived = p.derived();
    for (bi, b) in p.bodies.iter().enumerate() {
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        let m = &p.meshes[b.mesh as usize];
        let d = &derived[bi];
        for r in reflections(m.isx, m.isy) {
            for rec in 0..m.count as usize {
                if d.role[rec] == Role::Wet && d.tank[rec] < 0 {
                    walls.extend(edges_in_z0(m, rec, &r, &place));
                }
            }
        }
    }
    let found = free_edges_indexed(&plates, &walls, &DomainWalls::of(p), true);
    found.first().map(|&(ci, ri, _)| (plates[ci].label.clone(), plates[ci].recs[ri].number, plates[ci].body))
}

/// A plate of the sea over the fluid of a tank whose free surface lies in z = 0 (a floating roof, a
/// plate body of its own on another body's tank fluid): the plate and the tank in the messages and
/// the plate's body, else None. Such a plate would couple a body to the fluid of another body's tank; the plates of a
/// tank's free surface are the tank's own, moving with its body (theory §10.2). The test: the
/// winding of the plate element's centroid about the tank's waterline (the edges of its walls in
/// z = 0).
pub(crate) fn plate_on_tank(p: &Problem) -> Option<(String, String, i32)> {
    if !p.bodies.iter().any(|b| !b.plates.is_empty()) || !p.bodies.iter().any(|b| !b.tanks.is_empty()) {
        return None;
    }
    let cv = covers(p);
    let derived = p.derived();
    for (bi, b) in p.bodies.iter().enumerate() {
        let d = &derived[bi];
        let m = &p.meshes[b.mesh as usize];
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        for (ti, t) in b.tanks.iter().enumerate() {
            let zt = d.tank_z[ti] + b.position[2];
            if zt.abs() > 1e-6 * d.tank_z[ti].abs().max(b.position[2].abs()).max(1.0) {
                continue;
            }
            let mut edges: Vec<(P2, P2)> = Vec::new();
            for r in reflections(m.isx, m.isy) {
                for rec in 0..m.count as usize {
                    if d.tank[rec] == ti as i32 && d.role[rec] == Role::Tank && d.owner[rec] < 0 {
                        edges.extend(edges_in_z0(m, rec, &r, &place));
                    }
                }
            }
            let winding = Winding::new(&edges);
            for c in cv.iter().filter(|c| c.plate) {
                for e in &c.recs {
                    let n = e.boundary.len() as f64;
                    let x = e.boundary.iter().fold([0.0; 2], |s, q| [s[0] + q[0] / n, s[1] + q[1] / n]);
                    if winding.exceeds_pi(x) {
                        let tank =
                            format!("{} {}", body_label(b), element_text("tank", model_place(t.place, ti), &t.name));
                        return Some((c.label.clone(), tank, c.body));
                    }
                }
            }
        }
    }
    None
}

/// The rules: called by the rule check of a problem with free-surface plates, or with lids and
/// thin sheets (the junction of a sheet and a lid).
pub(crate) fn check(p: &Problem) -> Result<()> {
    let has_plates = p.bodies.iter().any(|b| !b.plates.is_empty());
    let has_sheets = p.bodies.iter().any(|b| b.has_sheets()) || !p.fixed_dampers.is_empty();
    if !has_plates && !(has_sheets && !p.lids.is_empty()) {
        return Ok(());
    }
    let cv = covers(p);
    if has_plates && let Some(&((x, rx), (y, ry))) = overlaps_indexed(&cv, true).first() {
        return Err(Error::value(format!(
            "{} overlaps {} in the free surface z = 0 (its element {} and element {}); a point of z = 0 belongs \
             to one surface only",
            cv[x].label, cv[y].label, cv[x].recs[rx].number, cv[y].recs[ry].number
        )));
    }
    if has_sheets {
        let edges = sheet_edges(p);
        let ends: Vec<(P2, P2)> = edges.iter().map(|e| (e.1, e.2)).collect();
        let columns: Vec<&Cover> = cv.iter().filter(|c| c.sheet_column).collect();
        if let Some((ei, ci, ri)) = sheet_inside_indexed(&ends, &columns) {
            let c = columns[ci];
            return Err(Error::value(format!(
                "{}, a thin sheet, meets {} inside its element {}: the potential jumps across the sheet, so the \
                 elements of {} must end on the sheet's edge in z = 0",
                edges[ei].0,
                c.label,
                c.recs[ri].number,
                if c.plate { "the plate" } else { "the lid" }
            )));
        }
    }
    Ok(())
}

/// The elements of the surfaces (cover, element) sorted by the lower x of their boxes (stable).
fn by_lower_x(cv: &[Cover]) -> Vec<(usize, usize)> {
    let mut items: Vec<(usize, usize)> =
        (0..cv.len()).flat_map(|c| (0..cv[c].recs.len()).map(move |r| (c, r))).collect();
    items.sort_by(|a, b| cv[a.0].recs[a.1].lo[0].total_cmp(&cv[b.0].recs[b.1].lo[0]));
    items
}

/// The overlaps wider than OVERLAP_WIDTH of an element of a plate and an element of another surface
/// of z = 0: (the plate's (cover, element), the other's), in the order of the pairs (i, j > i) of
/// `items` (by_lower_x), only the first when `first`; `later(i)`: the positions j > i of the
/// elements whose boxes may meet the box of item i, increasing (every one whose box meets it).
fn overlaps(
    cv: &[Cover],
    items: &[(usize, usize)],
    first: bool,
    later: &dyn Fn(usize) -> Vec<usize>,
) -> Vec<((usize, usize), (usize, usize))> {
    let mut out = Vec::new();
    for i in 0..items.len() {
        let (ci, ri) = items[i];
        let a = &cv[ci].recs[ri];
        for j in later(i) {
            let (cj, rj) = items[j];
            let b = &cv[cj].recs[rj];
            if ci == cj || !(cv[ci].plate || cv[cj].plate) || !boxes_meet(a, b) {
                continue;
            }
            if overlap(a, b) > OVERLAP_WIDTH {
                out.push(if cv[ci].plate { ((ci, ri), (cj, rj)) } else { ((cj, rj), (ci, ri)) });
                if first {
                    return out;
                }
            }
        }
    }
    out
}

/// The overlaps (overlaps) found through a tree of the elements' boxes: the same pairs, in the same
/// order, as the sweep along x of the elements sorted by lower x, in about n log n steps.
fn overlaps_indexed(cv: &[Cover], first: bool) -> Vec<((usize, usize), (usize, usize))> {
    let items = by_lower_x(cv);
    let tree = BoxTree::new(&items.iter().map(|&(c, r)| (cv[c].recs[r].lo, cv[c].recs[r].hi)).collect::<Vec<_>>());
    overlaps(cv, &items, first, &|i| {
        let a = &cv[items[i].0].recs[items[i].1];
        tree.search(a.lo, a.hi).into_iter().filter(|&j| j > i).collect()
    })
}

/// The overlaps (overlaps) by the sweep along x (each element against the later ones until their
/// lower x reaches its upper x): the reference of the indexed search in the tests.
#[cfg(test)]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the sweep's own stop (a NaN box does not stop it)
fn overlaps_swept(cv: &[Cover], first: bool) -> Vec<((usize, usize), (usize, usize))> {
    let items = by_lower_x(cv);
    overlaps(cv, &items, first, &|i| {
        let a = &cv[items[i].0].recs[items[i].1];
        (i + 1..items.len()).take_while(|&j| !(cv[items[j].0].recs[items[j].1].lo[0] >= a.hi[0])).collect()
    })
}

/// sheet_on_plate's and check's searches by a pass over every element: the references of the
/// indexed searches in the tests.
#[cfg(test)]
fn sheet_on_plate_brute(edges: &[(P2, P2)], plates: &[&Cover]) -> Option<(usize, usize)> {
    sheet_meeting(edges, plates.len(), &|c, x| plates[c].recs.iter().position(|r| on_element(r, x)))
        .map(|(e, c, _)| (e, c))
}

#[cfg(test)]
fn sheet_inside_brute(edges: &[(P2, P2)], columns: &[&Cover]) -> Option<(usize, usize, usize)> {
    sheet_meeting(edges, columns.len(), &|c, x| columns[c].recs.iter().position(|r| strictly_inside(r, x)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plate_geometry::tests::Rng;

    fn cover(recs: Vec<Rec>) -> Cover {
        Cover { label: String::new(), body: 0, plate: true, sheet_column: true, recs }
    }

    /// A grid of nx x ny quadrilaterals over [x0, x1] x [y0, y1], its inner nodes moved by up to
    /// `jitter` of a cell (conforming), each element of `split` x `split` pieces (a curved patch's
    /// pieces), counter-clockwise or not.
    #[allow(clippy::too_many_arguments)]
    fn grid(
        g: &mut Rng,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        nx: usize,
        ny: usize,
        jitter: f64,
        split: usize,
    ) -> Vec<Rec> {
        let (hx, hy) = ((x1 - x0) / nx as f64, (y1 - y0) / ny as f64);
        let mut node = vec![vec![[0.0; 2]; ny + 1]; nx + 1];
        for (i, col) in node.iter_mut().enumerate() {
            for (j, x) in col.iter_mut().enumerate() {
                let inner = i > 0 && i < nx && j > 0 && j < ny;
                let (dx, dy) =
                    if inner { (g.range(-jitter, jitter) * hx, g.range(-jitter, jitter) * hy) } else { (0.0, 0.0) };
                *x = [x0 + i as f64 * hx + dx, y0 + j as f64 * hy + dy];
            }
        }
        let mut out = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                let c = [node[i][j], node[i + 1][j], node[i + 1][j + 1], node[i][j + 1]];
                let at = |u: f64, v: f64| {
                    let (a, b) = (
                        [c[0][0] + u * (c[1][0] - c[0][0]), c[0][1] + u * (c[1][1] - c[0][1])],
                        [c[3][0] + u * (c[2][0] - c[3][0]), c[3][1] + u * (c[2][1] - c[3][1])],
                    );
                    [a[0] + v * (b[0] - a[0]), a[1] + v * (b[1] - a[1])]
                };
                let s = |k: usize| k as f64 / split as f64;
                let mut pieces = Vec::new();
                for a in 0..split {
                    for b in 0..split {
                        pieces.push(vec![
                            at(s(a), s(b)),
                            at(s(a + 1), s(b)),
                            at(s(a + 1), s(b + 1)),
                            at(s(a), s(b + 1)),
                        ]);
                    }
                }
                let mut boundary: Vec<P2> = (0..split).map(|k| at(s(k), 0.0)).collect();
                boundary.extend((0..split).map(|k| at(1.0, s(k))));
                boundary.extend((0..split).map(|k| at(1.0 - s(k), 1.0)));
                boundary.extend((0..split).map(|k| at(0.0, 1.0 - s(k))));
                if g.next() < 0.5 {
                    boundary.reverse();
                }
                out.push(rec_of(pieces, boundary, out.len() as i32 + 1));
            }
        }
        out
    }

    /// Walls along the outline of [x0, x1] x [y0, y1]: each side kept with probability `keep`, cut
    /// into a random number of chords moved off the side by up to `off` of their length.
    #[allow(clippy::too_many_arguments)]
    fn walls_around(g: &mut Rng, x0: f64, x1: f64, y0: f64, y1: f64, keep: f64, off: f64, out: &mut Vec<(P2, P2)>) {
        let c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        for k in 0..4 {
            if g.next() >= keep {
                continue;
            }
            let (a, b) = (c[k], c[(k + 1) % 4]);
            let n = 1 + (g.next() * 40.0) as usize;
            let len = (b[0] - a[0]).hypot(b[1] - a[1]) / n as f64;
            let (nx, ny) = ((b[1] - a[1]) / (n as f64 * len), -(b[0] - a[0]) / (n as f64 * len));
            let mut prev = a;
            for i in 1..=n {
                let t = i as f64 / n as f64;
                let e = g.range(-off, off) * len;
                let q = if i == n { b } else { [a[0] + t * (b[0] - a[0]) + e * nx, a[1] + t * (b[1] - a[1]) + e * ny] };
                out.push((prev, q));
                prev = q;
            }
        }
    }

    /// The indexed search of the free plate edges finds the same edges, in the same order, as the pass
    /// over every piece and every wall: conforming grids (jittered, of curved-patch pieces), grids
    /// side by side of other divisions (non-conforming, of the same body or of another), with gaps
    /// and overlaps of about the tenth of an edge, triangles at random, walls along their outlines
    /// off by about WALL_TOL of the plate edges, and the domain's walls.
    #[test]
    fn indexed_free_edges_are_those_of_the_brute_force() {
        let mut g = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut free, mut bordered, mut walled, mut first_some) = (0usize, 0usize, 0usize, 0usize);
        for case in 0..300 {
            let mut covers = Vec::new();
            let mut walls = Vec::new();
            let scale = 10f64.powf(g.range(-2.0, 3.0));
            let (cx, cy) = (g.range(-2.0, 2.0) * scale, g.range(-2.0, 2.0) * scale);
            let split = if g.next() < 0.3 { 2 } else { 1 };
            let (nx, ny) = (1 + (g.next() * 12.0) as usize, 1 + (g.next() * 12.0) as usize);
            let (w, h) = (scale * g.range(0.5, 2.0), scale * g.range(0.5, 2.0));
            let jitter = g.range(0.0, 0.3);
            covers.push(cover(grid(&mut g, cx, cx + w, cy, cy + h, nx, ny, jitter, split)));
            match case % 4 {
                // one conforming grid
                0 => {}
                // a second grid of another division beside it: non-conforming, touching, apart or
                // overlapping by about a tenth of an edge
                1 | 2 => {
                    let gap = if case % 4 == 1 { 0.0 } else { g.range(-0.2, 0.2) * w / nx as f64 };
                    let (mx, my) = (1 + (g.next() * 12.0) as usize, 1 + (g.next() * 12.0) as usize);
                    let w2 = scale * g.range(0.5, 2.0);
                    // of the same body or of another (its edges against the first grid then free)
                    let mut second = cover(grid(&mut g, cx + w + gap, cx + w + gap + w2, cy, cy + h, mx, my, 0.0, 1));
                    second.body = (case / 4) % 2;
                    covers.push(second);
                    walls_around(&mut g, cx + w + gap, cx + w + gap + w2, cy, cy + h, 0.5, 0.15, &mut walls);
                }
                // triangles at random over the grid
                _ => {
                    let mut recs = Vec::new();
                    for k in 0..(g.next() * 60.0) as usize {
                        let p = [cx + g.range(0.0, w), cy + g.range(0.0, h)];
                        let r = scale * g.range(0.02, 0.3);
                        let t: Vec<P2> = (0..3)
                            .map(|_| {
                                let a = g.range(0.0, std::f64::consts::TAU);
                                [p[0] + r * a.cos(), p[1] + r * a.sin()]
                            })
                            .collect();
                        recs.push(rec_of(vec![t.clone()], t, k as i32 + 1));
                    }
                    covers.push(cover(recs));
                }
            }
            walls_around(&mut g, cx, cx + w, cy, cy + h, 0.6, 0.15, &mut walls);
            for _ in 0..(g.next() * 20.0) as usize {
                let a = [cx + g.range(-0.2, 1.2) * w, cy + g.range(-0.2, 1.2) * h];
                let l = scale * g.range(0.01, 1.0);
                let t = g.range(0.0, std::f64::consts::TAU);
                walls.push((a, [a[0] + l * t.cos(), a[1] + l * t.sin()]));
            }
            let domain = match case % 5 {
                0 => DomainWalls { x: true, y: false, half: f64::NAN },
                1 => DomainWalls { x: true, y: true, half: f64::NAN },
                2 => DomainWalls { x: false, y: false, half: cy + h },
                _ => DomainWalls { x: false, y: false, half: f64::NAN },
            };
            if case % 5 <= 1 {
                // the grid against the walls x = 0 (and y = 0)
                let dy = if case % 5 == 1 { cy } else { 0.0 };
                let shift = |p: &mut P2| {
                    p[0] -= cx;
                    p[1] -= dy;
                };
                for c in covers.iter_mut() {
                    for r in c.recs.iter_mut() {
                        r.boundary.iter_mut().chain(r.pieces.iter_mut().flatten()).for_each(&shift);
                        (r.lo, r.hi) = ([r.lo[0] - cx, r.lo[1] - dy], [r.hi[0] - cx, r.hi[1] - dy]);
                    }
                }
                for (a, b) in walls.iter_mut() {
                    shift(a);
                    shift(b);
                }
            }
            let plates: Vec<&Cover> = covers.iter().collect();
            let fast = free_edges_indexed(&plates, &walls, &domain, false);
            let slow = free_edges_brute(&plates, &walls, &domain, false);
            assert_eq!(fast, slow, "case {case}");
            assert_eq!(
                free_edges_indexed(&plates, &walls, &domain, true),
                free_edges_brute(&plates, &walls, &domain, true),
                "case {case}"
            );
            let edges: usize = covers.iter().flat_map(|c| c.recs.iter()).map(|r| r.boundary.len()).sum();
            free += slow.len();
            walled += free_edges_brute(&plates, &[], &domain, false).len() - slow.len();
            bordered += edges - slow.len();
            first_some += usize::from(!slow.is_empty());
        }
        // both kinds of edges (bordered by a wall, not only by plate elements), and problems with and
        // without a free edge
        assert!(
            free > 1000 && bordered > 10000 && walled > 1000 && first_some > 50 && first_some < 300,
            "{free} {bordered} {walled} {first_some}"
        );
    }

    /// Random triangles of sizes `r0` to `r1` over [x0, x0 + w] x [y0, y0 + h].
    #[allow(clippy::too_many_arguments)]
    fn triangles(g: &mut Rng, n: usize, x0: f64, y0: f64, w: f64, h: f64, r0: f64, r1: f64) -> Vec<Rec> {
        (0..n)
            .map(|k| {
                let p = [x0 + g.range(0.0, w), y0 + g.range(0.0, h)];
                let r = g.range(r0, r1);
                let t: Vec<P2> = (0..3)
                    .map(|_| {
                        let a = g.range(0.0, std::f64::consts::TAU);
                        [p[0] + r * a.cos(), p[1] + r * a.sin()]
                    })
                    .collect();
                rec_of(vec![t.clone()], t, k as i32 + 1)
            })
            .collect()
    }

    /// The surfaces of z = 0 of a random problem at a random scale and place: a plate grid (jittered,
    /// of curved-patch pieces or not), then grids of other divisions (non-conforming) beside it,
    /// touching it or apart (every third problem: no overlap), or overlapping it by up to a third of
    /// its width, and random triangles; each a plate, a lid or another surface (an interior or
    /// pressure surface) at random. (covers, scale).
    fn random_surfaces(g: &mut Rng, case: usize) -> (Vec<Cover>, f64) {
        let scale = 10f64.powf(g.range(-2.0, 3.0));
        let (cx, cy) = (g.range(-2.0, 2.0) * scale, g.range(-2.0, 2.0) * scale);
        let (w, h) = (scale * g.range(0.5, 2.0), scale * g.range(0.5, 2.0));
        let mut out = Vec::new();
        let kind = |g: &mut Rng, recs: Vec<Rec>| {
            let u = g.next();
            Cover { label: String::new(), body: 0, plate: u < 0.5, sheet_column: u < 0.8, recs }
        };
        let split = if g.next() < 0.3 { 2 } else { 1 };
        let (nx, ny) = (1 + (g.next() * 10.0) as usize, 1 + (g.next() * 10.0) as usize);
        let jitter = g.range(0.0, 0.3);
        let first = grid(g, cx, cx + w, cy, cy + h, nx, ny, jitter, split);
        out.push(Cover { label: String::new(), body: 0, plate: true, sheet_column: true, recs: first });
        for _ in 0..1 + case % 3 {
            let shift = if case % 3 == 0 {
                w * if g.next() < 0.5 { 1.0 } else { g.range(1.0, 1.3) }
            } else {
                g.range(-0.35, 1.3) * w
            };
            let (mx, my) = (1 + (g.next() * 10.0) as usize, 1 + (g.next() * 10.0) as usize);
            let (x0, y0) = (cx + shift, cy + g.range(-0.3, 0.3) * h);
            let jitter = g.range(0.0, 0.2);
            let recs = grid(g, x0, x0 + w, y0, y0 + h, mx, my, jitter, 1);
            out.push(kind(g, recs));
        }
        if case % 2 == 1 && case % 3 != 0 {
            let n = (g.next() * 40.0) as usize;
            let recs = triangles(g, n, cx, cy, w, h, 0.02 * scale, 0.3 * scale);
            out.push(kind(g, recs));
        }
        (out, scale)
    }

    /// The indexed search of the overlaps of a plate with another surface finds the same pairs, in
    /// the same order (and so the same first pair), as the sweep along x: plate grids against grids
    /// of other divisions overlapping them, touching them or apart, plates, lids and other surfaces,
    /// random triangles, scales 1e-2 to 1e3.
    #[test]
    fn indexed_overlaps_are_those_of_the_sweep() {
        let mut g = Rng(0x6a09_e667_f3bc_c909);
        let (mut pairs, mut some) = (0usize, 0usize);
        for case in 0..200 {
            let (cv, _) = random_surfaces(&mut g, case);
            let slow = overlaps_swept(&cv, false);
            assert_eq!(overlaps_indexed(&cv, false), slow, "case {case}");
            assert_eq!(overlaps_indexed(&cv, true), overlaps_swept(&cv, true), "case {case}");
            pairs += slow.len();
            some += usize::from(!slow.is_empty());
        }
        assert!(pairs > 2000 && some > 50 && some < 180, "{pairs} {some}");
    }

    /// The indexed searches of a sheet's edge on a plate (sheet_on_plate: on an element or within
    /// EDGE_TOL of its edges) and inside an element of a plate or a lid (check: away from its
    /// edges) find the same first edge, surface (and element, inside) as the passes over every
    /// element: each edge alone and all of them, for edges along the elements' sides moved off them
    /// by up to twice EDGE_TOL of the element's diameter, edges across the surfaces at random and
    /// edges outside them.
    #[test]
    fn indexed_sheet_meetings_are_those_of_the_brute_force() {
        let mut g = Rng(0xbb67_ae85_84ca_a73b);
        let (mut on, mut inside, mut off) = (0usize, 0usize, 0usize);
        for case in 0..150 {
            let (cv, scale) = random_surfaces(&mut g, case);
            let plates: Vec<&Cover> = cv.iter().filter(|c| c.plate).collect();
            let columns: Vec<&Cover> = cv.iter().filter(|c| c.sheet_column).collect();
            let all: Vec<&Rec> = cv.iter().flat_map(|c| c.recs.iter()).collect();
            let (lo, hi) = (
                all.iter().fold([f64::INFINITY; 2], |s, r| [s[0].min(r.lo[0]), s[1].min(r.lo[1])]),
                all.iter().fold([f64::NEG_INFINITY; 2], |s, r| [s[0].max(r.hi[0]), s[1].max(r.hi[1])]),
            );
            let mut edges: Vec<(P2, P2)> = Vec::new();
            for _ in 0..60 {
                let u = g.next();
                if u < 0.6 {
                    // a side of an element, moved off it
                    let r = all[(g.next() * all.len() as f64) as usize];
                    let k = (g.next() * r.boundary.len() as f64) as usize;
                    let (a, b) = (r.boundary[k], r.boundary[(k + 1) % r.boundary.len()]);
                    let e = if g.next() < 0.3 { 0.0 } else { g.range(-2.0, 2.0) * EDGE_TOL * r.size };
                    let l = (b[0] - a[0]).hypot(b[1] - a[1]).max(f64::MIN_POSITIVE);
                    let n = [(b[1] - a[1]) / l * e, -(b[0] - a[0]) / l * e];
                    edges.push(([a[0] + n[0], a[1] + n[1]], [b[0] + n[0], b[1] + n[1]]));
                } else {
                    // across or outside the surfaces
                    let at = |g: &mut Rng| {
                        [
                            g.range(lo[0] - 0.3 * scale, hi[0] + 0.3 * scale),
                            g.range(lo[1] - 0.3 * scale, hi[1] + 0.3 * scale),
                        ]
                    };
                    let a = at(&mut g);
                    let b =
                        if u < 0.8 { at(&mut g) } else { [a[0] + 0.01 * scale, a[1] + g.range(-0.01, 0.01) * scale] };
                    edges.push((a, b));
                }
            }
            for k in 0..edges.len() {
                let one = &edges[k..k + 1];
                let fast = sheet_on_plate_indexed(one, &plates).map(|(e, c, _)| (e, c));
                assert_eq!(fast, sheet_on_plate_brute(one, &plates), "case {case} edge {k}");
                let fast_inside = sheet_inside_indexed(one, &columns);
                assert_eq!(fast_inside, sheet_inside_brute(one, &columns), "case {case} edge {k}");
                on += usize::from(fast.is_some());
                inside += usize::from(fast_inside.is_some());
                off += usize::from(fast.is_none() && fast_inside.is_none());
            }
            assert_eq!(
                sheet_on_plate_indexed(&edges, &plates).map(|(e, c, _)| (e, c)),
                sheet_on_plate_brute(&edges, &plates),
                "case {case}"
            );
            assert_eq!(sheet_inside_indexed(&edges, &columns), sheet_inside_brute(&edges, &columns), "case {case}");
        }
        assert!(on > 1000 && inside > 1000 && off > 1000, "{on} {inside} {off}");
    }
}
