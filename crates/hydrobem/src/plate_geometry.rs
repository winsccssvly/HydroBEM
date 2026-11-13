//! The geometry of the free-surface plates and the other surfaces in the plane z = 0: an element's
//! horizontal polygon (its convex pieces, boundary, box and diameter), the plane vertices of a mesh
//! (the footprint of a control surface, theory §11.3), the plate elements of a problem in the
//! numbering of the body outputs with the search of the plate beside a waterline side
//! (`PlateIndex`: the pressure drifts of both orders and the potential QTF), and the winding of a
//! point about a waterline (`Winding`). The rules on these surfaces are in plate_rules.rs.

use crate::constants::PI;
use crate::geometry::{Placement, Reflection, Vec3, reflections, v3};
use crate::mesh::{Mesh, patch_eval};
use crate::mesh_desc::PatchKind;
use crate::polygon::{BoxTree, P2, bounds, region_distance, signed_area};
use crate::problem::Problem;

/// One element of a surface in z = 0: convex pieces (the element itself, or a curved patch cut into
/// 8 x 8 pieces), its boundary, its box and diameter, all horizontal and global.
pub(crate) struct Rec {
    pub(crate) pieces: Vec<Vec<P2>>,
    pub(crate) boundary: Vec<P2>,
    pub(crate) lo: P2,
    pub(crate) hi: P2,
    pub(crate) size: f64,
    pub(crate) number: i32, // one-based, of its mesh as given
}

fn ccw(mut l: Vec<P2>) -> Vec<P2> {
    l.dedup_by(|a, b| (a[0] - b[0]).hypot(a[1] - b[1]) == 0.0);
    while l.len() > 1 && l[0] == l[l.len() - 1] {
        l.pop();
    }
    if signed_area(&l) < 0.0 {
        l.reverse();
    }
    l
}

pub(crate) fn rec_of(pieces: Vec<Vec<P2>>, boundary: Vec<P2>, number: i32) -> Rec {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &boundary {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let size = (hi[0] - lo[0]).hypot(hi[1] - lo[1]);
    Rec { pieces: pieces.into_iter().map(ccw).filter(|p| p.len() >= 3).collect(), boundary, lo, hi, size, number }
}

/// The global horizontal polygon of a point map of the parameter square of a patch element.
fn patch_points(m: &Mesh, rec: usize, f: &dyn Fn(Vec3) -> P2, n: usize) -> Vec<Vec<P2>> {
    let mut g = vec![vec![[0.0; 2]; n + 1]; n + 1];
    for (j, row) in g.iter_mut().enumerate() {
        for (i, x) in row.iter_mut().enumerate() {
            let (u, w) = (-1.0 + 2.0 * i as f64 / n as f64, -1.0 + 2.0 * j as f64 / n as f64);
            let (mut p, mut pu, mut pv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            patch_eval(m, rec as i32, u, w, &mut p, &mut pu, &mut pv);
            *x = f(p);
        }
    }
    g
}

/// Element rec of mesh m reflected by r and placed, as a Rec (number: its one-based number).
pub(crate) fn record(m: &Mesh, rec: usize, r: &Reflection, place: &Placement, number: i32) -> Rec {
    let f = |x: Vec3| {
        let y = place.apply(v3(r.sx * x[0], r.sy * x[1], x[2]));
        [y[0], y[1]]
    };
    if !m.patch || (m.geometry == PatchKind::Flat && !m.trimmed()) {
        let v: Vec<P2> = if m.patch {
            (0..4)
                .map(|k| {
                    f(v3(
                        m.vertices[12 * rec + 3 * k],
                        m.vertices[12 * rec + 3 * k + 1],
                        m.vertices[12 * rec + 3 * k + 2],
                    ))
                })
                .collect()
        } else {
            m.shapes[rec].hull.iter().map(|x| f(*x)).collect()
        };
        return rec_of(vec![v.clone()], v, number);
    }
    let n = 8;
    let g = patch_points(m, rec, &f, n);
    let mut pieces = Vec::new();
    for j in 0..n {
        for i in 0..n {
            pieces.push(vec![g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]]);
        }
    }
    let mut boundary: Vec<P2> = (0..n).map(|i| g[0][i]).collect();
    boundary.extend((0..n).map(|j| g[j][n]));
    boundary.extend((0..n).map(|i| g[n][n - i]));
    boundary.extend((0..n).map(|j| g[n - j][0]));
    rec_of(pieces, boundary, number)
}

/// The horizontal outline of element rec of mesh m in its own axes, every reflection of its
/// symmetry flags (the free-surface plates that an automatic control surface leaves out of its
/// free surface).
pub(crate) fn element_outlines(m: &Mesh, rec: usize) -> Vec<Vec<P2>> {
    let place = Placement { position: Vec3::ZERO, yaw_deg: 0.0 };
    reflections(m.isx, m.isy).iter().map(|r| record(m, rec, r, &place, 0).boundary).collect()
}

/// The vertices of a mesh's elements (its symmetry expanded) placed with body bi (none: global),
/// in the plane; of a patch mesh the points of each patch on its boundary, 16 a side, and its
/// centre (a generated control surface has patches only): the polygon of the samples lies
/// within the sag of a sixteenth of a side inside the footprint (1.2e-3 of the radius for a
/// cylinder of four patches of 90 degrees), any patch kind (record_eval: an untrimmed flat
/// patch, which has no spline data, by its bilinear map). The plane footprint of a control
/// surface (its convex hull, theory §11.3).
pub(crate) fn plane_vertices(p: &Problem, mesh: usize, bi: Option<usize>) -> Vec<[f64; 2]> {
    let ms = &p.meshes[mesh];
    let mut pts: Vec<Vec3> = Vec::new();
    if ms.patch {
        const N: usize = 16;
        let s = |k: usize| -1.0 + 2.0 * k as f64 / N as f64;
        let uv: Vec<(f64, f64)> = (0..N)
            .map(|k| (s(k), -1.0))
            .chain((0..N).map(|k| (1.0, s(k))))
            .chain((0..N).map(|k| (-s(k), 1.0)))
            .chain((0..N).map(|k| (-1.0, -s(k))))
            .chain(std::iter::once((0.0, 0.0)))
            .collect();
        for k in 0..ms.count {
            for &(u, v) in &uv {
                let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                crate::mesh::record_eval(ms, k, u, v, &mut x, &mut xu, &mut xv);
                pts.push(x);
            }
        }
    } else {
        pts = ms.panels.iter().flat_map(|q| q.raw.iter().copied()).collect();
    }
    let place = |x: f64, y: f64| match bi {
        Some(k) => {
            let b = &p.bodies[k];
            let (c, s) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
            [b.position[0] + c * x - s * y, b.position[1] + s * x + c * y]
        }
        None => [x, y],
    };
    let mut out = Vec::new();
    for rf in crate::geometry::reflections(ms.isx, ms.isy) {
        for v in &pts {
            out.push(place(rf.sx * v[0], rf.sy * v[1]));
        }
    }
    out
}

pub(crate) fn ranges_records(sel: &[crate::mesh::Range]) -> Vec<i32> {
    sel.iter().flat_map(|x| x.first - 1..x.last).collect()
}

/// An element of a free-surface plate in the plane z = 0 (global, every reflection of its mesh): its
/// body, its element as the body outputs number it (one-based: the elements of the body mesh, then
/// those of the meshes of its own of the body's pressure surfaces and plates, in that order), its
/// boundary and its longest side.
pub(crate) struct PlateElement {
    pub body: usize,
    pub element: i32,
    pub boundary: Vec<P2>,
    pub side: f64,
}

/// The tree of the boxes of plate elements (their boundaries and longest sides) grown by a tenth of
/// the longest side: the search of a point's nearest plate element within the larger of a radius w
/// and that tenth (PlateIndex: the plate beside a waterline side of the pressure drifts and the
/// potential QTF) is the search of the box of the point grown by w; its items
/// are those of the elements, in increasing order, so that the nearest one is found as by a pass
/// over all of them.
fn plate_tree<'a>(elements: impl Iterator<Item = (&'a [P2], f64)>) -> BoxTree {
    let boxes: Vec<(P2, P2)> = elements
        .map(|(l, side)| {
            let ((lo, hi), g) = (bounds(l), 0.1 * side);
            ([lo[0] - g, lo[1] - g], [hi[0] + g, hi[1] + g])
        })
        .collect();
    BoxTree::new(&boxes)
}

/// A plate element as the searches of PlateIndex see it: its horizontal boundary (global) and its
/// longest side.
pub(crate) trait PlateOutline {
    fn boundary(&self) -> &[P2];
    fn side(&self) -> f64;
}

impl PlateOutline for PlateElement {
    fn boundary(&self) -> &[P2] {
        &self.boundary
    }

    fn side(&self) -> f64 {
        self.side
    }
}

/// An element of a solver's own list: (its payload, its boundary, its longest side).
impl<T> PlateOutline for (T, Vec<P2>, f64) {
    fn boundary(&self) -> &[P2] {
        &self.1
    }

    fn side(&self) -> f64 {
        self.2
    }
}

/// Which of the candidates at the same smallest distance PlateIndex::beside returns: the first or
/// the last in the order of the elements. The higher-order pressure drift takes the last, the
/// low-order drifts and the potential QTF the first (their results are kept to the bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tie {
    First,
    Last,
}

/// The plate elements of a problem with the tree of their boxes (plate_tree): the one search of the
/// plate over the water beside a waterline side, made by the pressure drift of the higher order, the
/// drifts of the low order and the potential QTF.
pub(crate) struct PlateIndex<E> {
    pub(crate) elements: Vec<E>,
    tree: BoxTree,
}

impl<E: PlateOutline> PlateIndex<E> {
    pub(crate) fn new(elements: Vec<E>) -> Self {
        let tree = plate_tree(elements.iter().map(|e| (e.boundary(), e.side())));
        PlateIndex { elements, tree }
    }

    /// The plate element over the water beside a waterline side of midpoint `midpoint`, horizontal
    /// normal `normal` into the water (any length; none below 1e-6) and length `length`: the
    /// nearest element (`tie` among equal distances) to the point a twentieth of the length from the
    /// midpoint into the water, within a fortieth of the length or a tenth of the element's longest
    /// side (the chords of two meshes of one curve). None without one.
    pub(crate) fn beside(&self, midpoint: P2, normal: P2, length: f64, tie: Tie) -> Option<usize> {
        let h = normal[0].hypot(normal[1]);
        if self.elements.is_empty() || h < 1e-6 {
            return None;
        }
        let d = 0.05 * length;
        let x = [midpoint[0] + d * normal[0] / h, midpoint[1] + d * normal[1] / h];
        let w = 0.5 * d;
        let (mut best, mut at) = (f64::INFINITY, None);
        for i in self.tree.search([x[0] - w, x[1] - w], [x[0] + w, x[1] + w]) {
            let e = &self.elements[i];
            let r = region_distance(e.boundary(), x);
            let nearer = match tie {
                Tie::First => r < best,
                Tie::Last => r <= best,
            };
            if r <= w.max(0.1 * e.side()) && nearer {
                (best, at) = (r, Some(i));
            }
        }
        at
    }

    /// The first element (in their order) that holds the point x (its region distance zero).
    pub(crate) fn holding(&self, x: P2) -> Option<usize> {
        self.tree.search(x, x).into_iter().find(|&i| region_distance(self.elements[i].boundary(), x) == 0.0)
    }
}

/// The first element of every body's own meshes of pressure surfaces and plates, in the numbering
/// of the body outputs (after the body mesh's): (mesh, its first element less one).
fn own_meshes(p: &Problem, bi: usize) -> Vec<(i32, i32)> {
    let b = &p.bodies[bi];
    let mut next = p.meshes[b.mesh as usize].count;
    let mut out = Vec::new();
    for om in b.pressure.iter().filter_map(|x| x.mesh).chain(b.plates.iter().filter_map(|x| x.mesh)) {
        out.push((om as i32, next));
        next += p.meshes[om].count;
    }
    out
}

/// The elements of the free-surface plates of every body (PlateElement).
pub(crate) fn plate_elements(p: &Problem) -> Vec<PlateElement> {
    let mut out = Vec::new();
    for (bi, b) in p.bodies.iter().enumerate() {
        let place = Placement { position: b.position, yaw_deg: b.yaw };
        let own = own_meshes(p, bi);
        for pl in &b.plates {
            let (mi, recs, first) = if let Some(pi) = pl.mesh {
                let first = own.iter().find(|o| o.0 as usize == pi).map_or(0, |o| o.1);
                (pi, (0..p.meshes[pi].count).collect::<Vec<i32>>(), first)
            } else {
                (b.mesh as usize, ranges_records(&pl.panels), 0)
            };
            let m = &p.meshes[mi];
            for r in reflections(m.isx, m.isy) {
                for &rec in &recs {
                    let x = record(m, rec as usize, &r, &place, 0);
                    let l = &x.boundary;
                    let side = (0..l.len())
                        .map(|k| {
                            let (a, c) = (l[k], l[(k + 1) % l.len()]);
                            (c[0] - a[0]).hypot(c[1] - a[1])
                        })
                        .fold(0.0, f64::max);
                    out.push(PlateElement { body: bi, element: first + rec + 1, boundary: x.boundary, side });
                }
            }
        }
    }
    out
}

/// Whether the one-based element `element` of body bi's outputs (PlateElement) is an element of
/// one of its free-surface plates.
pub(crate) fn is_plate_element(p: &Problem, bi: usize, element: i32) -> bool {
    let b = &p.bodies[bi];
    if b.plates.is_empty() {
        return false;
    }
    let own = own_meshes(p, bi);
    b.plates.iter().any(|pl| {
        if let Some(pi) = pl.mesh {
            own.iter()
                .find(|o| o.0 as usize == pi)
                .is_some_and(|&(om, first)| element > first && element <= first + p.meshes[om as usize].count)
        } else {
            pl.panels.iter().any(|r| element >= r.first && element <= r.last)
        }
    })
}

/// The angle of the edge (a, c) seen from x, in (-pi, pi]: a term of the winding sum.
fn edge_angle(a: P2, c: P2, x: P2) -> f64 {
    let (ax, ay, cx, cy) = (a[0] - x[0], a[1] - x[1], c[0] - x[0], c[1] - x[1]);
    (ax * cy - ay * cx).atan2(ax * cx + ay * cy)
}

/// The winding sum about x of the edges (the sum of their angles seen from x, in their order).
fn winding_sum(edges: &[(P2, P2)], x: P2) -> f64 {
    edges.iter().map(|&(a, c)| edge_angle(a, c, x)).sum()
}

/// The distance from pi within which an estimate of the winding sum leaves the decision to the sum
/// itself: far above the difference of the two (rounding, in a sum of other order and of other
/// terms of the far edges), far below the distance of a sum about a point off the edges of closed
/// loops (a multiple of 2 pi) from pi.
const WINDING_MARGIN: f64 = 1e-3;

/// The decision |winding_sum| > pi about many points in about log n steps each (a hierarchy of the
/// edges' boxes, as Barnes and Hut 1986 sum a far field). Seen from a point x outside a node's box
/// grown by its diagonal d (at least d from the box, at least d from its centre, so the box, inside
/// the disc of radius d / 2 about its centre, lies in a cone of half-angle 30 degrees), the angle of
/// every edge of the node is the difference of the directions of its end points measured in that
/// cone, and their sum is that of the node's net end points (the ends less the starts at the same
/// point: none along the inside of a chain, the two ends of a chain) with their signs. The terms of
/// the other edges are those of the sum. The estimate differs from the sum by rounding; within
/// WINDING_MARGIN of pi the sum decides.
pub(crate) struct Winding<'a> {
    edges: &'a [(P2, P2)],
    tree: BoxTree,
    /// per node: its net end points (the point, +1 an end, -1 a start, the sum over its edges)
    charges: Vec<Vec<(P2, i32)>>,
    /// a coordinate not finite: the sum decides
    brute: bool,
}

impl<'a> Winding<'a> {
    pub(crate) fn new(edges: &'a [(P2, P2)]) -> Self {
        let brute =
            edges.iter().any(|(a, c)| !(a[0].is_finite() && a[1].is_finite() && c[0].is_finite() && c[1].is_finite()));
        let boxes: Vec<(P2, P2)> = if brute {
            Vec::new()
        } else {
            edges.iter().map(|&(a, c)| ([a[0].min(c[0]), a[1].min(c[1])], [a[0].max(c[0]), a[1].max(c[1])])).collect()
        };
        let tree = BoxTree::new(&boxes);
        // the same point under == (0.0 and -0.0 alike): the same bits once 0.0 is added
        let key = |p: P2| ((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits());
        let charges = (0..tree.node_count())
            .map(|n| {
                let mut ends: Vec<(P2, i32)> = Vec::new();
                for &i in tree.node(n).2 {
                    let (a, c) = edges[i];
                    ends.push(([a[0] + 0.0, a[1] + 0.0], -1));
                    ends.push(([c[0] + 0.0, c[1] + 0.0], 1));
                }
                ends.sort_unstable_by_key(|e| key(e.0));
                let mut net: Vec<(P2, i32)> = Vec::new();
                for (p, q) in ends {
                    match net.last_mut() {
                        Some(last) if key(last.0) == key(p) => last.1 += q,
                        _ => net.push((p, q)),
                    }
                }
                net.retain(|e| e.1 != 0);
                net
            })
            .collect();
        Winding { edges, tree, charges, brute }
    }

    /// The estimate of winding_sum(edges, x) (x finite).
    fn estimate(&self, x: P2) -> f64 {
        let mut s = 0.0;
        let mut stack = if self.tree.node_count() > 0 { vec![0] } else { Vec::new() };
        while let Some(n) = stack.pop() {
            let (lo, hi, items, children) = self.tree.node(n);
            let d = (hi[0] - lo[0]).hypot(hi[1] - lo[1]);
            if x[0] < lo[0] - d || x[0] > hi[0] + d || x[1] < lo[1] - d || x[1] > hi[1] + d {
                let r = [0.5 * (lo[0] + hi[0]) - x[0], 0.5 * (lo[1] + hi[1]) - x[1]];
                for &(p, q) in &self.charges[n] {
                    let v = [p[0] - x[0], p[1] - x[1]];
                    s += q as f64 * (r[0] * v[1] - r[1] * v[0]).atan2(r[0] * v[0] + r[1] * v[1]);
                }
            } else if let Some((l, r)) = children {
                stack.extend([r, l]);
            } else {
                s += items.iter().map(|&i| edge_angle(self.edges[i].0, self.edges[i].1, x)).sum::<f64>();
            }
        }
        s
    }

    /// Whether |winding_sum(edges, x)| > pi.
    pub(crate) fn exceeds_pi(&self, x: P2) -> bool {
        if !self.brute && x[0].is_finite() && x[1].is_finite() {
            let s = self.estimate(x).abs();
            if (s - std::f64::consts::PI).abs() > WINDING_MARGIN {
                return s > std::f64::consts::PI;
            }
        }
        winding_sum(self.edges, x).abs() > std::f64::consts::PI
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A xorshift generator of uniform numbers in [0, 1).
    pub(crate) struct Rng(pub(crate) u64);

    impl Rng {
        pub(crate) fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        pub(crate) fn range(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.next()
        }
    }

    /// The nearest plate element to a point within the larger of a radius and a tenth of the
    /// element's longest side (body_sides, the waterline of the low-order drifts) through plate_tree
    /// is the one of the pass over every element: quadrilaterals and triangles at random, of sizes
    /// over two decades, points at random, the radii from 0 to the elements' size.
    #[test]
    fn nearest_plate_elements_through_the_tree() {
        let mut g = Rng(0x2545_f491_4f6c_dd1d);
        let (mut found, mut none) = (0usize, 0usize);
        for _ in 0..50 {
            let scale = 10f64.powf(g.range(-2.0, 3.0));
            let elements: Vec<(Vec<P2>, f64)> = (0..(1.0 + g.next() * 400.0) as usize)
                .map(|_| {
                    let p = [g.range(0.0, 10.0) * scale, g.range(0.0, 10.0) * scale];
                    let r = scale * 10f64.powf(g.range(-1.5, 0.5));
                    let n = if g.next() < 0.5 { 3 } else { 4 };
                    let t0 = g.range(0.0, std::f64::consts::TAU);
                    let l: Vec<P2> = (0..n)
                        .map(|k| {
                            let t = t0 + std::f64::consts::TAU * (k as f64 + g.range(0.0, 0.5)) / n as f64;
                            [p[0] + r * t.cos(), p[1] + r * t.sin()]
                        })
                        .collect();
                    let side = (0..n).map(|k| (l[(k + 1) % n][0] - l[k][0]).hypot(l[(k + 1) % n][1] - l[k][1]));
                    let side = side.fold(0.0, f64::max);
                    (l, side)
                })
                .collect();
            let tree = plate_tree(elements.iter().map(|(l, s)| (l.as_slice(), *s)));
            for _ in 0..400 {
                let x = [g.range(-1.0, 11.0) * scale, g.range(-1.0, 11.0) * scale];
                let w = scale * g.range(0.0, 1.0) * g.next();
                let nearest = |cands: &mut dyn Iterator<Item = usize>| {
                    let (mut best, mut at) = (f64::INFINITY, None);
                    for i in cands {
                        let (l, side) = &elements[i];
                        let r = crate::polygon::region_distance(l, x);
                        if r <= w.max(0.1 * side) && r < best {
                            (best, at) = (r, Some(i));
                        }
                    }
                    at
                };
                let fast = nearest(&mut tree.search([x[0] - w, x[1] - w], [x[0] + w, x[1] + w]).into_iter());
                let slow = nearest(&mut (0..elements.len()));
                assert_eq!(fast, slow);
                if slow.is_some() {
                    found += 1;
                } else {
                    none += 1;
                }
            }
        }
        assert!(found > 2000 && none > 2000, "{found} {none}");
    }

    /// The three searches of the plate beside a waterline side as they were written in their
    /// solvers before PlateIndex (the higher-order pressure drift's plate_at, the low-order drifts'
    /// waterline_plates, the potential QTF's body_sides with its own conditions on the side), each
    /// returning the element's position in the list: the references of PlateIndex::beside.
    fn plate_at_old(plates: &[(usize, Vec<P2>, f64)], tree: &BoxTree, x: P2, n: P2, seg: f64) -> Option<usize> {
        let hn = n[0].hypot(n[1]);
        if plates.is_empty() || hn < 1e-6 {
            return None;
        }
        let d = 0.05 * seg;
        let y = [x[0] + d * n[0] / hn, x[1] + d * n[1] / hn];
        let (mut best, mut at) = (f64::INFINITY, None);
        let w = 0.5 * d;
        for i in tree.search([y[0] - w, y[1] - w], [y[0] + w, y[1] + w]) {
            let (_, l, side) = &plates[i];
            let r = region_distance(l, y);
            if r <= w.max(0.1 * side) && r <= best {
                (best, at) = (r, Some(i));
            }
        }
        at
    }

    fn waterline_plates_old(plates: &[(usize, Vec<P2>, f64)], tree: &BoxTree, x0: P2, n: P2, dl: f64) -> Option<usize> {
        let h = n[0].hypot(n[1]);
        if plates.is_empty() || h < 1e-6 {
            return None;
        }
        let d = 0.05 * dl;
        let x = [x0[0] + d * n[0] / h, x0[1] + d * n[1] / h];
        let (mut best, mut at) = (f64::INFINITY, None);
        let w = 0.5 * d;
        for i in tree.search([x[0] - w, x[1] - w], [x[0] + w, x[1] + w]) {
            let (_, l, side) = &plates[i];
            let r = region_distance(l, x);
            if r <= (0.5 * d).max(0.1 * side) && r < best {
                (best, at) = (r, Some(i));
            }
        }
        at
    }

    fn body_sides_old(plates: &[(usize, Vec<P2>, f64)], tree: &BoxTree, global: [P2; 2], g: P2) -> Option<usize> {
        let (gx, gy) = (g[0], g[1]);
        let h = gx.hypot(gy);
        let len = (global[1][0] - global[0][0]).hypot(global[1][1] - global[0][1]);
        let mut plate = None;
        if h > 1e-6 && len > 0.0 {
            let d = 0.05 * len;
            let x =
                [0.5 * (global[0][0] + global[1][0]) + d * gx / h, 0.5 * (global[0][1] + global[1][1]) + d * gy / h];
            let mut best = f64::INFINITY;
            let w = 0.5 * d;
            for i in tree.search([x[0] - w, x[1] - w], [x[0] + w, x[1] + w]) {
                let (_, l, side) = &plates[i];
                let r = region_distance(l, x);
                if r <= (0.5 * d).max(0.1 * side) && r < best {
                    (best, plate) = (r, Some(i));
                }
            }
        }
        plate
    }

    /// PlateIndex::beside gives each solver's old search's element to the bit, with the tie order
    /// of each (the last of equal distances for the higher order, the first for the others): plate
    /// elements at random and conforming grids of squares, points into the water landing exactly on
    /// the grid's lines and corners (equal distances, zero, from two to four elements: the tie
    /// orders then differ), points at random, normals of any length and none, sides of no length.
    #[test]
    fn plate_beside_a_side_is_that_of_each_old_search() {
        let mut g = Rng(0x510e_527f_ade6_82d1);
        let (mut found, mut ties, mut none) = (0usize, 0usize, 0usize);
        for case in 0..60 {
            let scale = 10f64.powf(g.range(-2.0, 3.0));
            let mut plates: Vec<(usize, Vec<P2>, f64)> = Vec::new();
            // a conforming grid of squares of side `a` from the origin (exact corners)
            let a = scale * [0.25, 0.5, 1.0, 2.0][case % 4];
            let (nx, ny) = (1 + (g.next() * 8.0) as usize, 1 + (g.next() * 8.0) as usize);
            for i in 0..nx {
                for j in 0..ny {
                    let (x0, y0) = (i as f64 * a, j as f64 * a);
                    let mut l = vec![[x0, y0], [x0 + a, y0], [x0 + a, y0 + a], [x0, y0 + a]];
                    if g.next() < 0.5 {
                        l.reverse();
                    }
                    plates.push((plates.len(), l, a));
                }
            }
            // random triangles and quadrilaterals over and around it
            for _ in 0..(g.next() * 80.0) as usize {
                let p = [g.range(-1.0, nx as f64 + 1.0) * a, g.range(-1.0, ny as f64 + 1.0) * a];
                let r = a * 10f64.powf(g.range(-1.5, 0.3));
                let n = if g.next() < 0.5 { 3 } else { 4 };
                let t0 = g.range(0.0, std::f64::consts::TAU);
                let l: Vec<P2> = (0..n)
                    .map(|k| {
                        let t = t0 + std::f64::consts::TAU * (k as f64 + g.range(0.0, 0.5)) / n as f64;
                        [p[0] + r * t.cos(), p[1] + r * t.sin()]
                    })
                    .collect();
                let side = (0..n)
                    .fold(0.0, |s: f64, k| s.max((l[(k + 1) % n][0] - l[k][0]).hypot(l[(k + 1) % n][1] - l[k][1])));
                plates.push((plates.len(), l, side));
            }
            if case % 10 == 9 {
                plates.clear();
            }
            let tree = plate_tree(plates.iter().map(|(_, l, s)| (l.as_slice(), *s)));
            let index = PlateIndex::new(plates.clone());
            for _ in 0..500 {
                let u = g.next();
                let len = if g.next() < 0.05 { 0.0 } else { a * g.range(0.1, 4.0) };
                let (mid, n) = if u < 0.5 {
                    // into the water along an axis onto a grid line or corner: the point lands on it
                    let (ix, iy) = ((g.next() * (nx + 1) as f64) as usize, (g.next() * (ny + 1) as f64) as usize);
                    let k = g.range(0.2, 3.0);
                    let s = if g.next() < 0.5 { -1.0 } else { 1.0 };
                    let d = 0.05 * len;
                    if g.next() < 0.5 {
                        ([ix as f64 * a, (iy as f64 + g.range(0.0, 1.0)) * a], [0.0, s * k])
                    } else {
                        ([ix as f64 * a - s * d, iy as f64 * a], [s * k, 0.0])
                    }
                } else {
                    let t = g.range(0.0, std::f64::consts::TAU);
                    let k = if g.next() < 0.05 { 1e-7 } else { g.range(0.1, 3.0) };
                    let p = [g.range(-1.0, nx as f64 + 1.0) * a, g.range(-1.0, ny as f64 + 1.0) * a];
                    (p, [k * t.cos(), k * t.sin()])
                };
                let last = index.beside(mid, n, len, Tie::Last);
                let first = index.beside(mid, n, len, Tie::First);
                assert_eq!(last, plate_at_old(&plates, &tree, mid, n, len), "case {case}");
                assert_eq!(first, waterline_plates_old(&plates, &tree, mid, n, len), "case {case}");
                // body_sides: its side's ends about the midpoint (the same midpoint to the bit), its
                // own conditions on the normal and the length
                let half = [0.5 * len, 0.0];
                let global = [[mid[0] - half[0], mid[1]], [mid[0] + half[0], mid[1]]];
                let len_s = (global[1][0] - global[0][0]).hypot(global[1][1] - global[0][1]);
                let mid_s = [0.5 * (global[0][0] + global[1][0]), 0.5 * (global[0][1] + global[1][1])];
                let h = n[0].hypot(n[1]);
                let new_s = if h > 1e-6 && len_s > 0.0 { index.beside(mid_s, n, len_s, Tie::First) } else { None };
                assert_eq!(new_s, body_sides_old(&plates, &tree, global, n), "case {case}");
                found += usize::from(first.is_some());
                ties += usize::from(first != last);
                none += usize::from(first.is_none());
            }
        }
        assert!(found > 5000 && ties > 1000 && none > 3000, "{found} {ties} {none}");
    }

    /// The first element holding a point (PlateIndex::holding) is that of the pass over every
    /// element, on the grid's lines and corners (two to four elements hold it) and at random.
    #[test]
    fn plate_holding_a_point_is_that_of_the_brute_force() {
        let mut g = Rng(0x9b05_688c_2b3e_6c1f);
        let mut held = 0usize;
        for case in 0..30 {
            let a = 10f64.powf(g.range(-2.0, 3.0));
            let (nx, ny) = (1 + (g.next() * 8.0) as usize, 1 + (g.next() * 8.0) as usize);
            let mut plates: Vec<(usize, Vec<P2>, f64)> = Vec::new();
            for i in 0..nx {
                for j in 0..ny {
                    let (x0, y0) = (i as f64 * a, j as f64 * a);
                    plates.push((plates.len(), vec![[x0, y0], [x0 + a, y0], [x0 + a, y0 + a], [x0, y0 + a]], a));
                }
            }
            let index = PlateIndex::new(plates.clone());
            for _ in 0..300 {
                let x = if g.next() < 0.5 {
                    [(g.next() * (nx + 1) as f64).floor() * a, g.range(-0.5, ny as f64 + 0.5) * a]
                } else {
                    [g.range(-0.5, nx as f64 + 0.5) * a, g.range(-0.5, ny as f64 + 0.5) * a]
                };
                let slow = plates.iter().position(|(_, l, _)| region_distance(l, x) == 0.0);
                assert_eq!(index.holding(x), slow, "case {case}");
                held += usize::from(slow.is_some());
            }
        }
        assert!(held > 3000, "{held}");
    }

    /// The waterline of a random tank: closed loops (a quadrant's chords reflected into the four
    /// quadrants, so that 0.0 meets -0.0 on the planes of symmetry; circles; an L; an annulus of
    /// opposite loops; a loop twice), or not (a loop with a gap, a corner moved by rounding, random
    /// segments), in random order, at a random scale and place.
    fn random_waterline(g: &mut Rng, case: usize) -> (Vec<(P2, P2)>, P2, f64) {
        let scale = 10f64.powf(g.range(-2.0, 3.0));
        let o = [g.range(-50.0, 50.0) * scale, g.range(-50.0, 50.0) * scale];
        let mut loops: Vec<Vec<P2>> = Vec::new();
        let circle = |g: &mut Rng, r: f64, n: usize, c: P2| -> Vec<P2> {
            let t0 = g.range(0.0, 1.0);
            (0..n)
                .map(|k| {
                    let t = std::f64::consts::TAU * (k as f64 + t0) / n as f64;
                    [c[0] + r * t.cos(), c[1] + r * t.sin()]
                })
                .collect()
        };
        match case % 6 {
            0 => {
                // a rectangle's quadrant in chords, reflected about x = 0 and y = 0 (placed at o by
                // the edges' shift below)
                let (a, b) = (scale * g.range(0.5, 2.0), scale * g.range(0.5, 2.0));
                let (na, nb) = (1 + (g.next() * 30.0) as usize, 1 + (g.next() * 30.0) as usize);
                let mut q: Vec<P2> = (0..=nb).map(|k| [a, b * k as f64 / nb as f64]).collect();
                q.extend((1..=na).map(|k| [a * (1.0 - k as f64 / na as f64), b]));
                let mut edges = Vec::new();
                for (sx, sy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                    for k in 0..q.len() - 1 {
                        let (p0, p1) = ([sx * q[k][0], sy * q[k][1]], [sx * q[k + 1][0], sy * q[k + 1][1]]);
                        edges.push(if sx * sy > 0.0 { (p0, p1) } else { (p1, p0) });
                    }
                }
                return (edges, [0.0, 0.0], scale);
            }
            1 => {
                let n = 3 + (g.next() * 200.0) as usize;
                loops.push(circle(g, scale, n, o));
            }
            2 => {
                let (a, b) = (scale, scale * g.range(0.2, 0.8));
                loops.push(vec![
                    o,
                    [o[0] + a, o[1]],
                    [o[0] + a, o[1] + b],
                    [o[0] + b, o[1] + b],
                    [o[0] + b, o[1] + a],
                    [o[0], o[1] + a],
                ]);
            }
            3 => {
                let n = 3 + (g.next() * 100.0) as usize;
                loops.push(circle(g, scale, n, o));
                let r = scale * g.range(0.2, 0.8);
                let mut inner = circle(g, r, n, o);
                inner.reverse();
                loops.push(inner);
            }
            4 => {
                let n = 3 + (g.next() * 60.0) as usize;
                let l = circle(g, scale, n, o);
                loops.push(l.clone());
                loops.push(if g.next() < 0.5 { l } else { l.into_iter().rev().collect() });
            }
            _ => {
                let mut edges: Vec<(P2, P2)> = Vec::new();
                let n = 3 + (g.next() * 60.0) as usize;
                let l = circle(g, scale, n, o);
                for k in 0..l.len() {
                    edges.push((l[k], l[(k + 1) % l.len()]));
                }
                match (g.next() * 3.0) as usize {
                    0 => {
                        edges.remove((g.next() * edges.len() as f64) as usize);
                    }
                    1 => {
                        let k = (g.next() * edges.len() as f64) as usize;
                        edges[k].1[0] += 1e-15 * scale.max(o[0].abs());
                    }
                    _ => {
                        for _ in 0..(g.next() * 40.0) as usize {
                            let a = [o[0] + g.range(-1.5, 1.5) * scale, o[1] + g.range(-1.5, 1.5) * scale];
                            let t = g.range(0.0, std::f64::consts::TAU);
                            let r = scale * g.range(0.01, 1.0);
                            edges.push((a, [a[0] + r * t.cos(), a[1] + r * t.sin()]));
                        }
                    }
                }
                return (edges, o, scale);
            }
        }
        let mut edges = Vec::new();
        for l in &loops {
            for k in 0..l.len() {
                edges.push((l[k], l[(k + 1) % l.len()]));
            }
        }
        // the elements' order, not the loop's
        for k in (1..edges.len()).rev() {
            edges.swap(k, (g.next() * (k + 1) as f64) as usize);
        }
        (edges, o, scale)
    }

    /// The decision |winding sum| > pi of the tree (Winding) is that of the sum over every edge:
    /// random waterlines (random_waterline) and points at random over and around them, far from
    /// them, exactly on their corners and the midpoints of their edges, and off their edges by 1e-12
    /// to 1e-3 of an edge. Most are decided by the estimate, not the sum.
    #[test]
    fn winding_through_the_tree_is_that_of_the_sum() {
        let mut g = Rng(0x3c6e_f372_fe94_f82b);
        let (mut yes, mut no, mut estimated) = (0usize, 0usize, 0usize);
        for case in 0..300 {
            let (edges, o, scale) = random_waterline(&mut g, case);
            let w = Winding::new(&edges);
            for _ in 0..300 {
                let u = g.next();
                let (a, c) = edges[(g.next() * edges.len() as f64) as usize];
                let x = if u < 0.5 {
                    [o[0] + g.range(-2.5, 2.5) * scale, o[1] + g.range(-2.5, 2.5) * scale]
                } else if u < 0.6 {
                    [o[0] + g.range(-1e3, 1e3) * scale, o[1] + g.range(-1e3, 1e3) * scale]
                } else if u < 0.7 {
                    a
                } else if u < 0.8 {
                    [0.5 * (a[0] + c[0]), 0.5 * (a[1] + c[1])]
                } else {
                    let t = g.next();
                    let e = 10f64.powf(g.range(-12.0, -3.0)) * if g.next() < 0.5 { -1.0 } else { 1.0 };
                    let (dx, dy) = (c[0] - a[0], c[1] - a[1]);
                    [a[0] + t * dx + e * dy, a[1] + t * dy - e * dx]
                };
                let slow = winding_sum(&edges, x).abs() > std::f64::consts::PI;
                assert_eq!(w.exceeds_pi(x), slow, "case {case} at {x:?}");
                if (w.estimate(x).abs() - std::f64::consts::PI).abs() > WINDING_MARGIN {
                    estimated += 1;
                }
                if slow {
                    yes += 1;
                } else {
                    no += 1;
                }
            }
        }
        assert!(yes > 15000 && no > 30000 && estimated > 80000, "{yes} {no} {estimated}");
    }
}
