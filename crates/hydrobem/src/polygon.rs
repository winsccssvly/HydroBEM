//! Polygons in the plane (theory §14): the nesting of closed loops into regions with holes, a
//! test for loops that cross or touch, and the triangulation of a region with holes: every hole
//! bridged to the outer boundary from its rightmost vertex (after Eberly, "Triangulation by ear
//! clipping", 2002: not his ray cast, but the nearest outer vertex visible from it, preferably to
//! its right, visible past the outer loop and every hole), the bridged polygon cut into ears, then
//! Lawson flips to the constrained Delaunay triangulation (the polygon edges are never flipped;
//! every inner edge is flipped until none is illegal), and its Delaunay refinement to a size and a
//! smallest angle (Ruppert 1995, Shewchuk 2002).

use std::collections::HashMap;

use crate::error::{Error, Result};

/// A point of the plane.
pub type P2 = [f64; 2];

/// Twice the signed area of the triangle (o, a, b): positive when counter-clockwise.
pub fn orient(o: P2, a: P2, b: P2) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// The signed area of a closed loop: positive when counter-clockwise.
pub fn signed_area(l: &[P2]) -> f64 {
    let n = l.len();
    0.5 * (0..n).map(|k| l[k][0] * l[(k + 1) % n][1] - l[(k + 1) % n][0] * l[k][1]).sum::<f64>()
}

/// Whether x is inside the closed loop (crossing number).
pub fn inside(l: &[P2], x: P2) -> bool {
    let n = l.len();
    let mut c = false;
    for k in 0..n {
        let (p, q) = (l[k], l[(k + 1) % n]);
        if (p[1] > x[1]) != (q[1] > x[1]) && x[0] < p[0] + (x[1] - p[1]) / (q[1] - p[1]) * (q[0] - p[0]) {
            c = !c;
        }
    }
    c
}

/// Whether x is inside the convex polygon (counter-clockwise) or on its boundary: on the left of
/// every edge or on it.
pub fn inside_convex(poly: &[P2], x: P2) -> bool {
    let n = poly.len();
    (0..n).all(|k| orient(poly[k], poly[(k + 1) % n], x) >= 0.0)
}

/// The convex hull of points in the plane, counter-clockwise (monotone chain), without collinear
/// vertices; fewer than three distinct points are returned sorted.
pub fn convex_hull(mut p: Vec<P2>) -> Vec<P2> {
    p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut lower: Vec<P2> = Vec::new();
    for &x in &p {
        while lower.len() >= 2 && orient(lower[lower.len() - 2], lower[lower.len() - 1], x) <= 0.0 {
            lower.pop();
        }
        lower.push(x);
    }
    let mut upper: Vec<P2> = Vec::new();
    for &x in p.iter().rev() {
        while upper.len() >= 2 && orient(upper[upper.len() - 2], upper[upper.len() - 1], x) <= 0.0 {
            upper.pop();
        }
        upper.push(x);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn dist(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Whether the segments pq and ab cross or touch.
pub fn meet(p: P2, q: P2, a: P2, b: P2) -> bool {
    let (d1, d2, d3, d4) = (orient(a, b, p), orient(a, b, q), orient(p, q, a), orient(p, q, b));
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    let on = |o: P2, e: P2, x: P2, d: f64| {
        d == 0.0 && x[0] >= o[0].min(e[0]) && x[0] <= o[0].max(e[0]) && x[1] >= o[1].min(e[1]) && x[1] <= o[1].max(e[1])
    };
    on(a, b, p, d1) || on(a, b, q, d2) || on(p, q, a, d3) || on(p, q, b, d4)
}

/// The distance from x to the segment ab (the rules of the plates keep their own, rounded as
/// x - a - t e, plate_rules.rs).
pub fn segment_distance(x: P2, a: P2, b: P2) -> f64 {
    let (e, w) = ([b[0] - a[0], b[1] - a[1]], [x[0] - a[0], x[1] - a[1]]);
    let ee = e[0] * e[0] + e[1] * e[1];
    let t = if ee > 0.0 { ((w[0] * e[0] + w[1] * e[1]) / ee).clamp(0.0, 1.0) } else { 0.0 };
    dist(x, [a[0] + t * e[0], a[1] + t * e[1]])
}

/// The distance from x to the closed loop's region: zero inside, else to its nearest edge.
pub fn region_distance(l: &[P2], x: P2) -> f64 {
    if inside(l, x) {
        return 0.0;
    }
    let n = l.len();
    (0..n).map(|k| segment_distance(x, l[k], l[(k + 1) % n])).fold(f64::INFINITY, f64::min)
}

/// Whether two of the loops, or two edges of one loop that do not follow each other, cross or come
/// within tol of each other (an end of one edge within tol of the other edge).
pub fn loops_meet(loops: &[Vec<P2>], tol: f64) -> bool {
    struct E {
        l: usize,
        k: usize,
        a: P2,
        b: P2,
    }
    let mut edges: Vec<E> = Vec::new();
    for (l, lp) in loops.iter().enumerate() {
        for k in 0..lp.len() {
            edges.push(E { l, k, a: lp[k], b: lp[(k + 1) % lp.len()] });
        }
    }
    let lo = |e: &E, c: usize| e.a[c].min(e.b[c]) - tol;
    let hi = |e: &E, c: usize| e.a[c].max(e.b[c]) + tol;
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            let (e, f) = (&edges[i], &edges[j]);
            if lo(e, 0) > hi(f, 0) || lo(f, 0) > hi(e, 0) || lo(e, 1) > hi(f, 1) || lo(f, 1) > hi(e, 1) {
                continue;
            }
            if e.l == f.l {
                let n = loops[e.l].len();
                if n <= 3 || (e.k + 1) % n == f.k || (f.k + 1) % n == e.k {
                    continue; // neighbours share a vertex (a triangle: every pair)
                }
            }
            if meet(e.a, e.b, f.a, f.b)
                || segment_distance(e.a, f.a, f.b) <= tol
                || segment_distance(e.b, f.a, f.b) <= tol
                || segment_distance(f.a, e.a, e.b) <= tol
                || segment_distance(f.b, e.a, e.b) <= tol
            {
                return true;
            }
        }
    }
    false
}

/// The loop without its vertices that lie within tol of the line through their neighbours (a
/// straight run of chords: their triangles would have no area), at least three kept.
pub fn merge_collinear(l: &[P2], tol: f64) -> Vec<P2> {
    let mut c = l.to_vec();
    let mut changed = true;
    while changed && c.len() > 3 {
        changed = false;
        for i in 0..c.len() {
            let n = c.len();
            let (a, b, d) = (c[(i + n - 1) % n], c[i], c[(i + 1) % n]);
            if orient(a, b, d).abs() <= tol * dist(a, d) {
                c.remove(i);
                changed = true;
                break;
            }
        }
    }
    c
}

/// A region of the plane: its outer loop (counter-clockwise) less its holes (clockwise).
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub outer: Vec<P2>,
    pub holes: Vec<Vec<P2>>,
}

/// The regions of closed loops that neither cross nor touch: a loop inside an even number of the
/// others bounds a region (made counter-clockwise), one inside an odd number is a hole (made
/// clockwise) of the smallest loop around it (an island in a hole bounds a region again).
pub fn regions(loops: Vec<Vec<P2>>) -> Vec<Region> {
    let n = loops.len();
    let area: Vec<f64> = loops.iter().map(|l| signed_area(l).abs()).collect();
    // the loops around each loop (a vertex of it inside them: the loops do not touch)
    let around: Vec<Vec<usize>> =
        (0..n).map(|i| (0..n).filter(|&j| j != i && inside(&loops[j], loops[i][0])).collect()).collect();
    let mut out: Vec<Region> = Vec::new();
    let mut region_of = vec![usize::MAX; n];
    for i in 0..n {
        if around[i].len() % 2 == 0 {
            let mut l = loops[i].clone();
            if signed_area(&l) < 0.0 {
                l.reverse();
            }
            region_of[i] = out.len();
            out.push(Region { outer: l, holes: Vec::new() });
        }
    }
    for i in 0..n {
        if around[i].len() % 2 == 1 {
            // the smallest loop around it bounds its region
            let owner = *around[i]
                .iter()
                .min_by(|&&a, &&b| area[a].partial_cmp(&area[b]).unwrap_or(std::cmp::Ordering::Equal))
                .expect("a hole lies inside a loop");
            let mut l = loops[i].clone();
            if signed_area(&l) > 0.0 {
                l.reverse();
            }
            out[region_of[owner]].holes.push(l);
        }
    }
    out
}

// the holes merged into the outer loop by a bridge from the rightmost vertex of each hole (the
// rightmost hole first) to the nearest vertex visible from it, preferably to its right; visible: the
// bridge meets no edge of the polygon so far, of the hole or of the holes not yet merged
fn bridge(r: &Region, eps: f64) -> Result<Vec<P2>> {
    let mut poly = r.outer.clone();
    let mut holes = r.holes.clone();
    let right = |l: &Vec<P2>| l.iter().map(|x| x[0]).fold(f64::NEG_INFINITY, f64::max);
    holes.sort_by(|a, b| right(b).partial_cmp(&right(a)).unwrap_or(std::cmp::Ordering::Equal));
    for (hi, h) in holes.iter().enumerate() {
        let mi = (0..h.len())
            .max_by(|&a, &b| h[a][0].partial_cmp(&h[b][0]).unwrap_or(std::cmp::Ordering::Equal))
            .expect("a hole has vertices");
        let m = h[mi];
        let n = poly.len();
        let later = &holes[hi + 1..];
        let visible = |k: usize| {
            let p = poly[k];
            (0..n).all(|e| e == k || (e + 1) % n == k || !meet(m, p, poly[e], poly[(e + 1) % n]))
                && (0..h.len()).all(|e| e == mi || (e + 1) % h.len() == mi || !meet(m, p, h[e], h[(e + 1) % h.len()]))
                && later.iter().all(|l| (0..l.len()).all(|e| !meet(m, p, l[e], l[(e + 1) % l.len()])))
        };
        let nearest = |a: &usize, b: &usize| {
            dist(poly[*a], m).partial_cmp(&dist(poly[*b], m)).unwrap_or(std::cmp::Ordering::Equal)
        };
        let best = (0..n)
            .filter(|&k| poly[k][0] >= m[0] - eps && visible(k))
            .min_by(nearest)
            .or_else(|| (0..n).filter(|&k| visible(k)).min_by(nearest));
        let Some(p) = best else {
            return Err(Error::internal("polygon triangulation: a hole without a bridge"));
        };
        let mut next = Vec::with_capacity(n + h.len() + 2);
        next.extend_from_slice(&poly[..=p]);
        for k in 0..=h.len() {
            next.push(h[(mi + k) % h.len()]);
        }
        next.extend_from_slice(&poly[p..]);
        poly = next;
    }
    Ok(poly)
}

// ear clipping of a counter-clockwise (bridged) polygon: index triples; a corner whose sine is
// below 1e-12 (a straight corner, or the doubled corner of a bridge) bounds no area and is dropped;
// every pass clips or drops one corner or fails, so the loop ends
fn ears(poly: &[P2], eps: f64) -> Result<Vec<[usize; 3]>> {
    let straight = |a: P2, b: P2, c: P2| orient(a, b, c).abs() <= 1e-12 * dist(a, b) * dist(b, c);
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..poly.len()).collect();
    while idx.len() > 3 {
        let n = idx.len();
        let mut clipped = false;
        for i in 0..n {
            let (ia, ib, ic) = (idx[(i + n - 1) % n], idx[i], idx[(i + 1) % n]);
            let (a, b, c) = (poly[ia], poly[ib], poly[ic]);
            if straight(a, b, c) {
                idx.remove(i);
                clipped = true;
                break;
            }
            if orient(a, b, c) < 0.0 {
                continue; // reflex
            }
            let ear = idx.iter().all(|&k| {
                let x = poly[k];
                let at = |y: P2| (x[0] - y[0]).abs() <= eps && (x[1] - y[1]).abs() <= eps;
                at(a) || at(b) || at(c) || !(orient(a, b, x) > 0.0 && orient(b, c, x) > 0.0 && orient(c, a, x) > 0.0)
            });
            if ear {
                out.push([ia, ib, ic]);
                idx.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            return Err(Error::internal("polygon triangulation: the ear clipping failed"));
        }
    }
    if idx.len() == 3 && !straight(poly[idx[0]], poly[idx[1]], poly[idx[2]]) {
        out.push([idx[0], idx[1], idx[2]]);
    }
    Ok(out)
}

// Lawson flips to the constrained Delaunay triangulation: an inner edge (shared by two triangles;
// the polygon edges are in one) is illegal when the opposite vertex of one triangle lies in the
// circumcircle of the other by more than the rounding of the test and the two triangles form a
// convex quadrilateral; it is then flipped. Every pass visits every inner edge of the current
// triangles (at the later of its two triangles), the map of the edges to their triangles kept up
// to date at each flip, until a pass flips none: then no inner edge is illegal. Lawson's flips end
// (each one raises the triangulation in the lifting map); more than T² flips of T triangles would be
// a fault of the triangulation, an internal error.
fn delaunay_flips(poly: &[P2], tris: &mut [[usize; 3]]) -> Result<()> {
    let incircle = |a: usize, b: usize, c: usize, d: usize| {
        let (pa, pb, pc, pd) = (poly[a], poly[b], poly[c], poly[d]);
        let (adx, ady, bdx, bdy) = (pa[0] - pd[0], pa[1] - pd[1], pb[0] - pd[0], pb[1] - pd[1]);
        let (cdx, cdy) = (pc[0] - pd[0], pc[1] - pd[1]);
        (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
            + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    };
    const NONE: usize = usize::MAX;
    let key = |a: usize, b: usize| (a.min(b), a.max(b));
    // (min, max) -> its triangles; an edge in more than two (not a triangulation of a polygon) is
    // never flipped
    let mut owners: HashMap<(usize, usize), [usize; 2]> = HashMap::with_capacity(2 * tris.len());
    let mut fixed: Vec<(usize, usize)> = Vec::new();
    for (t, q) in tris.iter().enumerate() {
        for e in 0..3 {
            let o = owners.entry(key(q[e], q[(e + 1) % 3])).or_insert([NONE, NONE]);
            if o[0] == NONE {
                o[0] = t;
            } else if o[1] == NONE {
                o[1] = t;
            } else {
                fixed.push(key(q[e], q[(e + 1) % 3]));
            }
        }
    }
    for k in fixed {
        owners.insert(k, [NONE, NONE]);
    }
    let limit = tris.len().saturating_mul(tris.len()).max(16);
    let mut flips = 0usize;
    loop {
        let mut flipped = false;
        for t in 0..tris.len() {
            for e in 0..3 {
                let (a, b, c) = (tris[t][e], tris[t][(e + 1) % 3], tris[t][(e + 2) % 3]);
                let o = owners[&key(a, b)];
                let u = if o[0] == t { o[1] } else { o[0] };
                if u == NONE || u > t {
                    continue; // a polygon edge, or visited at its later triangle
                }
                let Some(f) = (0..3).find(|&f| tris[u][f] == b && tris[u][(f + 1) % 3] == a) else {
                    return Err(Error::internal("polygon triangulation: two triangles turned the same way"));
                };
                let d = tris[u][(f + 2) % 3];
                // the in-circle determinant (a length^4) above its rounding: 1e-12 of the square
                // of the triangle's orientation determinant (a length^2), the same test in any unit
                let ob = orient(poly[a], poly[b], poly[c]);
                if incircle(a, b, c, d) > 1e-12 * ob * ob
                    && orient(poly[c], poly[d], poly[b]) > 0.0
                    && orient(poly[d], poly[c], poly[a]) > 0.0
                {
                    tris[t] = [c, d, b];
                    tris[u] = [d, c, a];
                    owners.remove(&key(a, b));
                    owners.insert(key(c, d), [t, u]);
                    for (edge, from, to) in [(key(d, b), u, t), (key(c, a), t, u)] {
                        let o = owners.get_mut(&edge).expect("an edge of the quadrilateral");
                        for x in o.iter_mut() {
                            if *x == from {
                                *x = to;
                            }
                        }
                    }
                    flipped = true;
                    flips += 1;
                    if flips > limit {
                        return Err(Error::internal("polygon triangulation: the Delaunay flips do not end"));
                    }
                    break; // the triangle t is new: its edges at the next pass
                }
            }
        }
        if !flipped {
            return Ok(());
        }
    }
}

/// The triangles (counter-clockwise) of a region: its holes bridged to the outer loop, the ears of
/// the bridged polygon, then the Lawson flips. eps: the distance within which two vertices are one
/// (the two ends of a bridge).
pub fn triangulate(r: &Region, eps: f64) -> Result<Vec<[P2; 3]>> {
    let poly = bridge(r, eps)?;
    let mut t = ears(&poly, eps)?;
    delaunay_flips(&poly, &mut t)?;
    Ok(t.iter().map(|q| [poly[q[0]], poly[q[1]], poly[q[2]]]).collect())
}

// ------------------------------------------------------------------ Delaunay refinement

/// The smallest angle the refinement aims at (degrees): below the 33.8 degrees up to which
/// Delaunay refinement by circumcentres ends in practice, above the 20.7 of Ruppert's proof.
pub const REFINE_MIN_ANGLE: f64 = 25.0;

const NONE: usize = usize::MAX;

fn key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn incircle(pa: P2, pb: P2, pc: P2, pd: P2) -> f64 {
    let (adx, ady, bdx, bdy) = (pa[0] - pd[0], pa[1] - pd[1], pb[0] - pd[0], pb[1] - pd[1]);
    let (cdx, cdy) = (pc[0] - pd[0], pc[1] - pd[1]);
    (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
}

fn circumcentre(a: P2, b: P2, c: P2) -> P2 {
    let (bx, by, cx, cy) = (b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]);
    let d = 2.0 * (bx * cy - by * cx);
    let (b2, c2) = (bx * bx + by * by, cx * cx + cy * cy);
    [a[0] + (cy * b2 - by * c2) / d, a[1] + (bx * c2 - cx * b2) / d]
}

// a triangulation of a polygon's vertices: the triangles (counter-clockwise, dead ones kept as
// holes in the list), the map of each edge to its two triangles and the segments (the edges of
// the region's boundary, with the length of the boundary edge each was split from)
struct Mesh2 {
    p: Vec<P2>,
    t: Vec<[usize; 3]>,
    alive: Vec<bool>,
    owners: HashMap<(usize, usize), [usize; 2]>,
    segs: HashMap<(usize, usize), f64>,
}

enum Located {
    Inside(usize),
    Segment(usize, usize),
    Lost,
}

impl Mesh2 {
    fn add_tri(&mut self, q: [usize; 3]) -> usize {
        let id = self.t.len();
        self.t.push(q);
        self.alive.push(true);
        for e in 0..3 {
            let o = self.owners.entry(key(q[e], q[(e + 1) % 3])).or_insert([NONE, NONE]);
            if o[0] == NONE { o[0] = id } else { o[1] = id }
        }
        id
    }
    fn kill(&mut self, id: usize) {
        self.alive[id] = false;
        let q = self.t[id];
        for e in 0..3 {
            let k = key(q[e], q[(e + 1) % 3]);
            if let Some(o) = self.owners.get_mut(&k) {
                for x in o.iter_mut() {
                    if *x == id {
                        *x = NONE;
                    }
                }
                if o[0] == NONE && o[1] == NONE {
                    self.owners.remove(&k);
                }
            }
        }
    }
    // the triangle across the edge (a, b) of the triangle t
    fn across(&self, t: usize, a: usize, b: usize) -> usize {
        match self.owners.get(&key(a, b)) {
            Some(o) if o[0] == t => o[1],
            Some(o) => o[0],
            None => NONE,
        }
    }
    // the triangle that holds c, walked from the triangle t along the line from its centroid to c,
    // or the first segment the line meets
    fn locate(&self, t: usize, c: P2) -> Located {
        let q = self.t[t];
        let g = [
            (self.p[q[0]][0] + self.p[q[1]][0] + self.p[q[2]][0]) / 3.0,
            (self.p[q[0]][1] + self.p[q[1]][1] + self.p[q[2]][1]) / 3.0,
        ];
        let mut cur = t;
        let mut from = NONE; // the triangle the walk came from
        for _ in 0..self.t.len() + 3 {
            let q = self.t[cur];
            // the edges with c beyond them that the line crosses: the first not back to the last
            // triangle (a line through a vertex has two)
            let crossed: Vec<(usize, usize, usize)> = (0..3)
                .map(|e| (q[e], q[(e + 1) % 3]))
                .filter(|&(u, v)| {
                    let (su, sv) = (orient(g, c, self.p[u]), orient(g, c, self.p[v]));
                    orient(self.p[u], self.p[v], c) < 0.0 && ((su <= 0.0 && sv >= 0.0) || (su >= 0.0 && sv <= 0.0))
                })
                .map(|(u, v)| (u, v, self.across(cur, u, v)))
                .collect();
            let exit = crossed.iter().copied().find(|x| x.2 != from).or(crossed.first().copied());
            let Some((u, v, n)) = exit else {
                return if (0..3).all(|e| orient(self.p[q[e]], self.p[q[(e + 1) % 3]], c) >= 0.0) {
                    Located::Inside(cur)
                } else {
                    Located::Lost
                };
            };
            if self.segs.contains_key(&key(u, v)) || n == NONE {
                return Located::Segment(u, v);
            }
            from = cur;
            cur = n;
        }
        Located::Lost
    }
    // the cavity of the point x from the triangle t: the triangles reachable from t across edges
    // that are not segments whose circumcircles hold x (t always), and its boundary edges in the
    // sense of their triangles
    fn cavity(&self, t: usize, x: P2) -> (Vec<usize>, Vec<(usize, usize)>) {
        let mut cav = vec![t];
        let mut k = 0;
        while k < cav.len() {
            let q = self.t[cav[k]];
            for e in 0..3 {
                let (u, v) = (q[e], q[(e + 1) % 3]);
                if self.segs.contains_key(&key(u, v)) {
                    continue;
                }
                let n = self.across(cav[k], u, v);
                if n == NONE || cav.contains(&n) {
                    continue;
                }
                let r = self.t[n];
                if incircle(self.p[r[0]], self.p[r[1]], self.p[r[2]], x) > 0.0 {
                    cav.push(n);
                }
            }
            k += 1;
        }
        let mut boundary = Vec::new();
        for &c in &cav {
            let q = self.t[c];
            for e in 0..3 {
                let (u, v) = (q[e], q[(e + 1) % 3]);
                let n = self.across(c, u, v);
                if n == NONE || !cav.contains(&n) || self.segs.contains_key(&key(u, v)) {
                    boundary.push((u, v));
                }
            }
        }
        (cav, boundary)
    }
    // the cavity replaced by the fan from the new vertex x to its boundary edges but `skip`; None
    // (nothing changed) when a triangle of the fan would not turn counter-clockwise by more than
    // the rounding; else the new triangles
    fn insert(
        &mut self,
        cav: &[usize],
        boundary: &[(usize, usize)],
        x: P2,
        skip: Option<(usize, usize)>,
    ) -> Option<Vec<usize>> {
        let fan: Vec<(usize, usize)> = boundary.iter().copied().filter(|&(u, v)| Some((u, v)) != skip).collect();
        for &(u, v) in &fan {
            let (pu, pv) = (self.p[u], self.p[v]);
            let l2 = (pu[0] - pv[0]).powi(2) + (pu[1] - pv[1]).powi(2);
            if orient(pu, pv, x) <= 1e-9 * l2 {
                return None;
            }
        }
        let id = self.p.len();
        self.p.push(x);
        for &c in cav {
            self.kill(c);
        }
        Some(fan.iter().map(|&(u, v)| self.add_tri([u, v, id])).collect())
    }
}

/// The triangles (counter-clockwise) of a region refined to the size `size` and to the smallest
/// angle REFINE_MIN_ANGLE: the triangulation of `triangulate` (its bridges merged), then Delaunay
/// refinement (Ruppert 1995, with Shewchuk's rules, 2002): a triangle whose longest edge exceeds
/// `size`, or whose smallest angle is below REFINE_MIN_ANGLE, gets the vertex at its circumcentre
/// (Bowyer-Watson insertion, the triangles whose circumcircles hold it re-made, never across a
/// boundary edge); a boundary edge that the circumcentre encroaches upon (it lies in its diametral
/// circle) or lies beyond is split at its midpoint instead. A boundary edge is split only while
/// longer than `size` or than 3/4 of the smaller of `size`/2 and the boundary edge it was split from
/// (so every boundary edge at most once below the size), and a triangle is refined for its angle
/// only when its shortest edge is at least a quarter of the shortest boundary edge (the corners of
/// the boundary sharper than the angle stay so). The vertices of the boundary are kept, the new
/// ones on it lie on its edges: the region is the same. Stops at `cap` triangles. One sequence of
/// insertions in a fixed order: the same triangles on every run.
pub fn triangulate_refined(r: &Region, eps: f64, size: f64, cap: usize) -> Result<Vec<[P2; 3]>> {
    let poly = bridge(r, eps)?;
    let tris = ears(&poly, eps)?;
    // the two ends of each bridge are exact copies: one vertex
    let mut p: Vec<P2> = Vec::new();
    let mut id_of: HashMap<(u64, u64), usize> = HashMap::new();
    let ids: Vec<usize> = poly
        .iter()
        .map(|x| {
            *id_of.entry((x[0].to_bits(), x[1].to_bits())).or_insert_with(|| {
                p.push(*x);
                p.len() - 1
            })
        })
        .collect();
    let mut t: Vec<[usize; 3]> = tris.iter().map(|q| [ids[q[0]], ids[q[1]], ids[q[2]]]).collect();
    delaunay_flips(&p, &mut t)?;
    let mut m = Mesh2 { p, t: Vec::new(), alive: Vec::new(), owners: HashMap::new(), segs: HashMap::new() };
    for q in t {
        m.add_tri(q);
    }
    // the segments: the edges of one triangle
    let mut shortest = f64::INFINITY;
    for i in 0..m.t.len() {
        let q = m.t[i];
        for e in 0..3 {
            let (u, v) = (q[e], q[(e + 1) % 3]);
            if m.across(i, u, v) == NONE {
                let l = dist(m.p[u], m.p[v]);
                m.segs.insert(key(u, v), l);
                shortest = shortest.min(l);
            }
        }
    }
    let ratio = 0.5 / (REFINE_MIN_ANGLE * crate::constants::PI / 180.0).sin(); // circumradius / shortest edge
    let mut queue: std::collections::VecDeque<usize> = (0..m.t.len()).collect();
    let mut live = m.t.len();
    while let Some(i) = queue.pop_front() {
        if !m.alive[i] || live > cap {
            continue;
        }
        let q = m.t[i];
        let (a, b, c) = (m.p[q[0]], m.p[q[1]], m.p[q[2]]);
        let l = [dist(b, c), dist(c, a), dist(a, b)];
        let (lo, hi) = (l[0].min(l[1]).min(l[2]), l[0].max(l[1]).max(l[2]));
        let cc = circumcentre(a, b, c);
        let big = hi > size;
        let skinny = dist(cc, a) > ratio * lo && lo >= 0.25 * shortest;
        if !big && !skinny {
            continue;
        }
        let split_ok = |m: &Mesh2, s: (usize, usize)| {
            let l = dist(m.p[s.0], m.p[s.1]);
            l > size || l >= 0.75 * (0.5 * size).min(m.segs[&key(s.0, s.1)])
        };
        let mut fan = None;
        let split = match m.locate(i, cc) {
            Located::Lost => continue,
            Located::Segment(u, v) => Some((u, v)),
            Located::Inside(t0) => {
                let (cav, bnd) = m.cavity(t0, cc);
                // a boundary edge of the cavity whose diametral circle holds the circumcentre
                let encroached = bnd.iter().copied().find(|&(u, v)| {
                    m.segs.contains_key(&key(u, v)) && {
                        let (pu, pv) = (m.p[u], m.p[v]);
                        (pu[0] - cc[0]) * (pv[0] - cc[0]) + (pu[1] - cc[1]) * (pv[1] - cc[1]) < 0.0
                    }
                });
                if encroached.is_none() {
                    fan = m.insert(&cav, &bnd, cc, None).map(|new| (new, cav.len()));
                }
                encroached
            }
        };
        if let Some((u, v)) = split {
            if !split_ok(&m, (u, v)) {
                continue;
            }
            let s = m.owners[&key(u, v)];
            let ts = if s[0] != NONE { s[0] } else { s[1] };
            // the edge in the sense of its triangle
            let qs = m.t[ts];
            let (u, v) = (0..3)
                .map(|e| (qs[e], qs[(e + 1) % 3]))
                .find(|&(x, y)| key(x, y) == key(u, v))
                .expect("an edge of its triangle");
            let mid = [0.5 * (m.p[u][0] + m.p[v][0]), 0.5 * (m.p[u][1] + m.p[v][1])];
            let (cav, bnd) = m.cavity(ts, mid);
            let from = m.segs[&key(u, v)];
            if let Some(new) = m.insert(&cav, &bnd, mid, Some((u, v))) {
                let x = m.p.len() - 1;
                m.segs.remove(&key(u, v));
                m.segs.insert(key(u, x), from);
                m.segs.insert(key(x, v), from);
                live += new.len() - cav.len();
                queue.extend(new);
                if m.alive[i] {
                    queue.push_back(i);
                }
            }
        } else if let Some((new, removed)) = fan {
            live += new.len() - removed;
            queue.extend(new);
        }
    }
    Ok((0..m.t.len()).filter(|&i| m.alive[i]).map(|i| [m.p[m.t[i][0]], m.p[m.t[i][1]], m.p[m.t[i][2]]]).collect())
}

/// The bounding box of a loop: (its lower corner, its upper corner).
pub fn bounds(l: &[P2]) -> (P2, P2) {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in l {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

/// A static tree of boxes of the plane (a bounding-volume hierarchy: the boxes halved at the
/// median of their centres along the wider extent, down to leaves of a few boxes), which finds the
/// boxes that meet a query box in about log n + k steps instead of a pass over all n. It only
/// selects the candidates of an exact test made after it: every box is grown by 1e-9 of the scale
/// of the coordinates, so that no rounding of that test can accept an item the search left out.
pub struct BoxTree {
    lo: Vec<P2>,
    hi: Vec<P2>,
    order: Vec<usize>,
    nodes: Vec<BoxNode>,
}

/// A node: its box, its items order[first..last], its children (none for a leaf).
struct BoxNode {
    lo: P2,
    hi: P2,
    first: usize,
    last: usize,
    children: Option<(usize, usize)>,
}

const BOX_LEAF: usize = 8;

impl BoxTree {
    /// The tree of the boxes (lower corner, upper corner); item i is boxes[i].
    pub fn new(boxes: &[(P2, P2)]) -> Self {
        let scale = boxes.iter().flat_map(|(a, b)| [a[0], a[1], b[0], b[1]]).fold(0.0, |s: f64, x| s.max(x.abs()));
        let g = 1e-9 * scale.max(f64::MIN_POSITIVE);
        let lo: Vec<P2> = boxes.iter().map(|(a, _)| [a[0] - g, a[1] - g]).collect();
        let hi: Vec<P2> = boxes.iter().map(|(_, b)| [b[0] + g, b[1] + g]).collect();
        let mut t = BoxTree { lo, hi, order: (0..boxes.len()).collect(), nodes: Vec::new() };
        if !boxes.is_empty() {
            t.build(0, boxes.len());
        }
        t
    }

    fn build(&mut self, first: usize, last: usize) -> usize {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        let (mut clo, mut chi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for &i in &self.order[first..last] {
            for k in 0..2 {
                lo[k] = lo[k].min(self.lo[i][k]);
                hi[k] = hi[k].max(self.hi[i][k]);
                let c = 0.5 * (self.lo[i][k] + self.hi[i][k]);
                clo[k] = clo[k].min(c);
                chi[k] = chi[k].max(c);
            }
        }
        let at = self.nodes.len();
        self.nodes.push(BoxNode { lo, hi, first, last, children: None });
        if last - first > BOX_LEAF {
            let k = if chi[0] - clo[0] >= chi[1] - clo[1] { 0 } else { 1 };
            let mid = first + (last - first) / 2;
            let (blo, bhi) = (&self.lo, &self.hi);
            let centre = |i: usize| blo[i][k] + bhi[i][k];
            self.order[first..last].select_nth_unstable_by(mid - first, |&a, &b| centre(a).total_cmp(&centre(b)));
            let left = self.build(first, mid);
            let right = self.build(mid, last);
            self.nodes[at].children = Some((left, right));
        }
        at
    }

    /// Whether some item whose box meets the box (lo, hi) (edges included) satisfies f: the items
    /// are tried in no particular order, until f accepts one.
    pub fn any(&self, lo: P2, hi: P2, mut f: impl FnMut(usize) -> bool) -> bool {
        let meets = |a: P2, b: P2| a[0] <= hi[0] && lo[0] <= b[0] && a[1] <= hi[1] && lo[1] <= b[1];
        let mut stack = if self.nodes.is_empty() { Vec::new() } else { vec![0] };
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if !meets(node.lo, node.hi) {
                continue;
            }
            match node.children {
                Some((l, r)) => stack.extend([r, l]),
                None => {
                    for &i in &self.order[node.first..node.last] {
                        if meets(self.lo[i], self.hi[i]) && f(i) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// The number of nodes (none in a tree of no boxes; the root is node 0).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Node n: its box (holding its items' grown boxes), its items (in no particular order) and
    /// its children (none for a leaf), for a walk of the tree of one's own.
    pub fn node(&self, n: usize) -> (P2, P2, &[usize], Option<(usize, usize)>) {
        let x = &self.nodes[n];
        (x.lo, x.hi, &self.order[x.first..x.last], x.children)
    }

    /// The items whose boxes meet the box (lo, hi) (edges included), in increasing order.
    pub fn search(&self, lo: P2, hi: P2) -> Vec<usize> {
        let mut out = Vec::new();
        self.any(lo, hi, |i| {
            out.push(i);
            false
        });
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_area(t: &[[P2; 3]]) -> f64 {
        t.iter().map(|q| 0.5 * orient(q[0], q[1], q[2])).sum()
    }

    // a square with a square hole: counter-clockwise triangles whose areas sum to the region's
    // (16 - 4, exact to rounding), none over the hole
    #[test]
    fn square_with_a_hole() -> Result<()> {
        let outer = vec![[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]];
        let hole = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let rs = regions(vec![hole, outer]);
        assert_eq!(rs.len(), 1);
        assert_eq!(rs[0].holes.len(), 1);
        assert!(signed_area(&rs[0].outer) > 0.0 && signed_area(&rs[0].holes[0]) < 0.0);
        let t = triangulate(&rs[0], 1e-9)?;
        assert!(t.iter().all(|q| orient(q[0], q[1], q[2]) > 0.0));
        assert!((total_area(&t) - 12.0).abs() < 1e-12);
        for q in &t {
            let c = [(q[0][0] + q[1][0] + q[2][0]) / 3.0, (q[0][1] + q[1][1] + q[2][1]) / 3.0];
            assert!(c[0].abs() >= 1.0 || c[1].abs() >= 1.0, "a triangle over the hole");
        }
        Ok(())
    }

    // an island in the hole bounds a region of its own; two loops side by side are two regions
    #[test]
    fn nesting() {
        let sq = |a: f64, x0: f64| vec![[x0 - a, -a], [x0 + a, -a], [x0 + a, a], [x0 - a, a]];
        let rs = regions(vec![sq(3.0, 0.0), sq(2.0, 0.0), sq(1.0, 0.0)]);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs.iter().map(|r| r.holes.len()).sum::<usize>(), 1);
        let rs = regions(vec![sq(1.0, -3.0), sq(1.0, 3.0)]);
        assert_eq!(rs.len(), 2);
        assert!(rs.iter().all(|r| r.holes.is_empty()));
    }

    // a non-convex (C-shaped) polygon: covered once
    #[test]
    fn c_shape() -> Result<()> {
        let c = vec![[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [0.0, 3.0]];
        let t = triangulate(&Region { outer: c, holes: Vec::new() }, 1e-9)?;
        assert!((total_area(&t) - 7.0).abs() < 1e-12);
        assert!(t.iter().all(|q| orient(q[0], q[1], q[2]) > 0.0));
        Ok(())
    }

    // the constrained Delaunay property of triangles that tile a region: for every edge two
    // triangles share (each the other way round), the opposite vertex of the one is not inside the
    // circumcircle of the other by more than the rounding of the test, or the two do not form a
    // convex quadrilateral (the flip would not be allowed)
    fn delaunay(t: &[[P2; 3]]) -> bool {
        let incircle = |pa: P2, pb: P2, pc: P2, pd: P2| {
            let (adx, ady, bdx, bdy) = (pa[0] - pd[0], pa[1] - pd[1], pb[0] - pd[0], pb[1] - pd[1]);
            let (cdx, cdy) = (pc[0] - pd[0], pc[1] - pd[1]);
            (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
                + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
        };
        for (i, p) in t.iter().enumerate() {
            for (j, q) in t.iter().enumerate() {
                if i == j {
                    continue;
                }
                for e in 0..3 {
                    let (a, b, c) = (p[e], p[(e + 1) % 3], p[(e + 2) % 3]);
                    let Some(f) = (0..3).find(|&f| q[f] == b && q[(f + 1) % 3] == a) else {
                        continue;
                    };
                    let d = q[(f + 2) % 3];
                    let o = orient(a, b, c);
                    if incircle(a, b, c, d) > 1e-12 * o * o && orient(c, d, b) > 0.0 && orient(d, c, a) > 0.0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    // the flips make the triangulation Delaunay: a long thin convex polygon with 121 vertices on
    // each long side (bulged by 1e-3 so that no corner is straight), whose ears are a fan of slivers
    // from one corner (hundreds of flips, beyond the 200 of a fixed cap), and a square with a hole;
    // the triangles still tile the region
    #[test]
    fn the_flips_make_the_triangles_delaunay() -> Result<()> {
        let n = 120;
        let bulge = |k: usize| 1e-3 * (crate::constants::PI * k as f64 / n as f64).sin();
        let mut r: Vec<P2> = (0..=n).map(|k| [k as f64, -bulge(k)]).collect();
        r.extend((0..=n).rev().map(|k| [k as f64, 1.0 + bulge(k)]));
        let thin = Region { outer: r.clone(), holes: Vec::new() };
        // the ears alone are not Delaunay
        let poly = bridge(&thin, 1e-9)?;
        let ear: Vec<[P2; 3]> = ears(&poly, 1e-9)?.iter().map(|q| [poly[q[0]], poly[q[1]], poly[q[2]]]).collect();
        assert!(!delaunay(&ear));
        let t = triangulate(&thin, 1e-9)?;
        assert!(delaunay(&t));
        assert!((total_area(&t) - signed_area(&r)).abs() < 1e-9);
        assert_eq!(t.len(), r.len() - 2);
        assert!(t.iter().all(|q| orient(q[0], q[1], q[2]) > 0.0));
        let outer: Vec<P2> = (0..40).map(|k| [-2.0 + 0.1 * k as f64, -2.0]).collect::<Vec<_>>();
        let mut sq = outer.clone();
        sq.extend((0..40).map(|k| [2.0, -2.0 + 0.1 * k as f64]));
        sq.extend((0..40).map(|k| [2.0 - 0.1 * k as f64, 2.0]));
        sq.extend((0..40).map(|k| [-2.0, 2.0 - 0.1 * k as f64]));
        let hole = vec![[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]];
        let t = triangulate(&Region { outer: sq, holes: vec![hole] }, 1e-9)?;
        assert!(delaunay(&t));
        assert!((total_area(&t) - 15.0).abs() < 1e-9);
        Ok(())
    }

    // a bridge does not cross a hole not yet merged: three holes in a row, the rightmost bridged
    // first; the triangles tile the region (its area exactly), none over a hole
    #[test]
    fn bridges_pass_the_other_holes() -> Result<()> {
        let sq =
            |a: f64, x0: f64, y0: f64| vec![[x0 - a, y0 + a], [x0 + a, y0 + a], [x0 + a, y0 - a], [x0 - a, y0 - a]];
        let outer = vec![[-4.0, -2.0], [4.0, -2.0], [4.0, 2.0], [-4.0, 2.0]];
        let holes = vec![sq(0.5, -2.0, 0.0), sq(0.5, 0.0, 0.3), sq(0.5, 2.0, 0.0)];
        let t = triangulate(&Region { outer, holes: holes.clone() }, 1e-9)?;
        assert!((total_area(&t) - (32.0 - 3.0)).abs() < 1e-9);
        for q in &t {
            let c = [(q[0][0] + q[1][0] + q[2][0]) / 3.0, (q[0][1] + q[1][1] + q[2][1]) / 3.0];
            assert!(holes.iter().all(|h| !inside(h, c)), "a triangle over a hole");
        }
        assert!(delaunay(&t));
        Ok(())
    }

    // crossing loops, touching loops and a loop crossing itself meet; disjoint ones do not
    #[test]
    fn meeting_loops() {
        let sq = |a: f64, x0: f64| vec![[x0 - a, -a], [x0 + a, -a], [x0 + a, a], [x0 - a, a]];
        assert!(loops_meet(&[sq(1.0, 0.0), sq(1.0, 1.0)], 1e-9));
        assert!(loops_meet(&[sq(1.0, 0.0), sq(1.0, 2.0)], 1e-9));
        assert!(!loops_meet(&[sq(1.0, 0.0), sq(1.0, 3.0)], 1e-9));
        let bow = vec![[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        assert!(loops_meet(&[bow], 1e-9));
    }

    // the smallest angle of a triangle (degrees)
    fn min_angle(q: &[P2; 3]) -> f64 {
        (0..3)
            .map(|k| {
                let (a, b, c) = (q[k], q[(k + 1) % 3], q[(k + 2) % 3]);
                let (u, v) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
                (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1]).to_degrees()
            })
            .fold(180.0, f64::min)
    }

    // every edge of the triangles is the edge of one other triangle, the other way round, or lies
    // on the boundary of the region: no vertex on the edge of another triangle (a conforming mesh)
    fn conforming(t: &[[P2; 3]], on_boundary: impl Fn(P2, P2) -> bool) -> bool {
        let bits = |x: P2| (x[0].to_bits(), x[1].to_bits());
        type Bits = (u64, u64);
        let mut e: HashMap<(Bits, Bits), usize> = HashMap::new();
        for q in t {
            for k in 0..3 {
                *e.entry((bits(q[k]), bits(q[(k + 1) % 3]))).or_insert(0) += 1;
            }
        }
        t.iter().all(|q| {
            (0..3).all(|k| {
                let (a, b) = (q[k], q[(k + 1) % 3]);
                e.get(&(bits(b), bits(a))) == Some(&1) || on_boundary(a, b)
            })
        })
    }

    // the refinement of a quarter disc (16 chords and the two radii, as the waterline of a
    // quadrant), a quarter annulus (a moonpool), a C shape and a square with a hole: the region
    // exactly (its area), conforming, every edge within the size, the smallest angle at least 20
    // degrees; the same triangles on a second run; the triangulation alone has slivers
    #[test]
    fn the_refinement_meets_its_bounds() -> Result<()> {
        let arc = |r: f64, n: usize| -> Vec<P2> {
            (0..=n)
                .map(|k| {
                    let t = 0.5 * crate::constants::PI * k as f64 / n as f64;
                    [r * t.cos(), r * t.sin()]
                })
                .collect()
        };
        let mut quarter = vec![[0.0, 0.0]];
        quarter.extend(arc(1.0, 16));
        let mut annulus = arc(2.0, 16);
        annulus.extend(arc(1.0, 8).into_iter().rev());
        let c = vec![[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [0.0, 3.0]];
        let sq = vec![[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]];
        let hole = vec![[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]];
        let cases = [
            (Region { outer: quarter, holes: Vec::new() }, 0.2),
            (Region { outer: annulus, holes: Vec::new() }, 0.3),
            (Region { outer: c, holes: Vec::new() }, 0.5),
            (Region { outer: sq, holes: vec![hole] }, 0.6),
        ];
        for (r, size) in &cases {
            let t = triangulate_refined(r, 1e-9, *size, 100_000)?;
            let area = signed_area(&r.outer) + r.holes.iter().map(|h| signed_area(h)).sum::<f64>();
            assert!((total_area(&t) - area).abs() <= 1e-12 * area, "the region");
            assert!(t.iter().all(|q| orient(q[0], q[1], q[2]) > 0.0));
            let loops: Vec<&Vec<P2>> = std::iter::once(&r.outer).chain(r.holes.iter()).collect();
            let on_boundary = |a: P2, b: P2| {
                loops.iter().any(|l| {
                    (0..l.len()).any(|k| {
                        let (p, q) = (l[k], l[(k + 1) % l.len()]);
                        segment_distance(a, p, q) <= 1e-12 && segment_distance(b, p, q) <= 1e-12
                    })
                })
            };
            assert!(conforming(&t, on_boundary), "conforming");
            for q in &t {
                assert!((0..3).all(|k| dist(q[k], q[(k + 1) % 3]) <= size * (1.0 + 1e-12)), "the size");
            }
            let worst = t.iter().map(min_angle).fold(180.0, f64::min);
            assert!(worst >= 20.0, "the smallest angle {worst}");
            let again = triangulate_refined(r, 1e-9, *size, 100_000)?;
            assert!(t.len() == again.len() && t.iter().zip(&again).all(|(a, b)| a == b), "deterministic");
        }
        let t = triangulate(&cases[0].0, 1e-9)?;
        assert!(t.iter().map(min_angle).fold(180.0, f64::min) < 10.0);
        Ok(())
    }

    // a corner sharper than the angle (10 degrees) stays sharp and the refinement ends; it stops
    // at its cap
    #[test]
    fn the_refinement_ends_at_a_sharp_corner_and_at_its_cap() -> Result<()> {
        let s = 10f64.to_radians();
        let wedge = Region { outer: vec![[0.0, 0.0], [4.0, 0.0], [4.0 * s.cos(), 4.0 * s.sin()]], holes: Vec::new() };
        let t = triangulate_refined(&wedge, 1e-9, 0.5, 100_000)?;
        let area = signed_area(&wedge.outer);
        assert!((total_area(&t) - area).abs() <= 1e-12 * area);
        assert!(t.len() < 2000, "{}", t.len());
        let t = triangulate_refined(&wedge, 1e-9, 0.01, 50)?;
        assert!(t.len() > 50 && t.len() < 60, "{}", t.len());
        Ok(())
    }
}
