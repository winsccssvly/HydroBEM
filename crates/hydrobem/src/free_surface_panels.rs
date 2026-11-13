//! The panels of the free surface about the bodies for the potential QTF (theory §12.8): from
//! the waterlines to the circle of radius R about the centre, of about the panel size s.
//!
//! With r_c the largest distance of a waterline side's middle from the centre:
//! - a square of half-width b = √2 r_c, a uniform grid inside it of spacing b / ⌈b / √(s s_w)⌉
//!   (s_w the mean waterline side, the grid at most s): the cells clear of the waterlines are
//!   panels, the rest of the water inside the grid triangles between the cells and the waterlines
//!   (each region with its waterlines as holes triangulated by crate::polygon: bridges, ear
//!   clipping, Delaunay flips);
//! - the outermost row of the square: triangles between the grid's last line and the square's
//!   nodes on the spokes below;
//! - spokes: n_q a quadrant (the mean radius' quarter circumference over s, rounded down to ten
//!   when above ten, even), from the square to the circle r_0 = 2 r_c in ⌈(r_0 − b)/s⌉ layers of
//!   points blended linearly between the square and the circle (the square's corner lies on it),
//!   then rings r_0 .. R in ⌈(R − r_0)/s⌉ equal steps.
//!
//! Every panel is integrated by its centroid with its area. The panels tile the region exactly
//! (their areas sum to π R² less the waterlines' areas). Why they are the default free surface
//! (one point per panel against 4 × 4 per cell, the first-order field at the points being the
//! dominant cost): theory §12.8 and §12.12.

use crate::constants::PI;
use crate::error::{Error, Result};
use crate::polygon::{Region, inside, meet, signed_area as area, triangulate};

/// A panel: four corners counter-clockwise, a triangle with its third corner twice.
pub type Panel = [[f64; 2]; 4];

/// The closed loops of the segments (joined end to end within tol; each loop counter-clockwise).
fn loops(segs: &[([f64; 2], [f64; 2])], tol: f64) -> Result<Vec<Vec<[f64; 2]>>> {
    let same = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= tol;
    let mut used = vec![false; segs.len()];
    let mut out = Vec::new();
    for s0 in 0..segs.len() {
        if used[s0] || same(segs[s0].0, segs[s0].1) {
            used[s0] = true;
            continue;
        }
        used[s0] = true;
        let start = segs[s0].0;
        let mut lp = vec![start];
        let mut end = segs[s0].1;
        while !same(end, start) {
            let next = (0..segs.len()).find(|&k| !used[k] && (same(segs[k].0, end) || same(segs[k].1, end)));
            let Some(k) = next else {
                return Err(Error::value("second_order: a waterline is not closed (panels need closed waterlines)"));
            };
            used[k] = true;
            lp.push(end);
            end = if same(segs[k].0, end) { segs[k].1 } else { segs[k].0 };
        }
        if lp.len() >= 3 {
            if area(&lp) < 0.0 {
                lp.reverse();
            }
            out.push(lp);
        }
    }
    Ok(out)
}

/// The panels of the free surface from the waterline sides segs (global) to the circle of
/// radius r_out about c0, of about size s (the module's rule). Global coordinates.
pub fn panels(segs: &[([f64; 2], [f64; 2])], c0: [f64; 2], r_out: f64, s: f64) -> Result<Vec<Panel>> {
    let shift: Vec<([f64; 2], [f64; 2])> =
        segs.iter().map(|(p, q)| ([p[0] - c0[0], p[1] - c0[1]], [q[0] - c0[0], q[1] - c0[1]])).collect();
    if shift.is_empty() {
        return Err(Error::value("second_order: the bodies have no waterline"));
    }
    let rc = shift.iter().map(|(p, q)| (0.5 * (p[0] + q[0])).hypot(0.5 * (p[1] + q[1]))).fold(0.0, f64::max);
    let sw = shift.iter().map(|(p, q)| (q[0] - p[0]).hypot(q[1] - p[1])).sum::<f64>() / shift.len() as f64;
    let b = 2f64.sqrt() * rc;
    let r0 = 2.0 * rc;
    if r_out.is_nan() || r_out <= r0 * (1.0 + 1e-9) || s.is_nan() || s <= 0.0 {
        return Err(Error::value(format!(
            "second_order: the inner radius ({r_out} m) must exceed twice the largest distance of a waterline from the centre ({r0} m)"
        )));
    }
    // the closed waterlines; one inside another (a moonpool, or a body in a moonpool) is refused:
    // the panels cover the water outside the waterlines only, the water inside the inner one would
    // be left out of the integral (and the hole inside a hole is no simple polygon to triangulate)
    let wl = loops(&shift, 1e-6 * rc)?;
    for (i, a) in wl.iter().enumerate() {
        if let Some(x) = wl.iter().enumerate().find_map(|(j, o)| (j != i && inside(o, a[0])).then_some(a[0])) {
            return Err(Error::value(format!(
                "second_order: a waterline lies inside another (a moonpool), through ({}, {}) m: the generated panels cover the water outside the waterlines only (give a free-surface mesh that covers the water inside it too)",
                x[0] + c0[0],
                x[1] + c0[1]
            )));
        }
    }
    // the waterlines inside the grid's last line (half-width b − h) by a tenth of a step at least
    // (b − vmax > 1.1 h): the rows and rings outside it are not cut by the waterlines; a coarse
    // waterline (its corners far beyond the middles of its sides) takes more steps than the panel
    // size gives, at most 50 (or the steps of the panel size when more); one whose corners reach the
    // square none
    let vmax = shift.iter().flat_map(|(p, q)| [p, q]).map(|x| x[0].abs().max(x[1].abs())).fold(0.0, f64::max);
    let by_size = ((b / (s * sw).sqrt().min(s)).ceil() as usize).max(2);
    let fit = if vmax < b { (1.1 * b / (b - vmax)).floor() + 1.0 } else { f64::INFINITY }; // b − vmax > 1.1 b / n
    if fit > by_size.max(50) as f64 {
        return Err(Error::value(format!(
            "second_order: a waterline corner lies {vmax} m from the centre along an axis, too near the square of the generated panels (half-width {b} m, √2 times the largest distance of a waterline side's middle): the waterline is too coarse for the generated panels (give finer waterline sides, or a free-surface mesh)"
        )));
    }
    let n = by_size.max(fit as usize);
    let h = b / n as f64;
    let eps = 1e-9 * b;
    // the spokes
    let raw = 0.5 * (r0 + r_out) * 0.5 * PI / s;
    let mut nq = if raw >= 10.0 { (raw / 10.0).floor() as usize * 10 } else { raw.ceil() as usize };
    nq = nq.max(2);
    if nq % 2 == 1 {
        nq += 1;
    }
    let ns = 4 * nq;
    let th = |j: usize| 0.5 * PI * j as f64 / nq as f64;
    let foot = |j: usize| {
        let (c, sn) = (th(j).cos(), th(j).sin());
        let f = b / c.abs().max(sn.abs());
        [f * c, f * sn]
    };
    let mut out: Vec<Panel> = Vec::new();
    let quad = |a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]| [a, b, c, d];
    // the rings r0 .. R
    let lp = ((r_out - r0) / s).ceil().max(1.0) as usize;
    let ring = |k: usize, j: usize| {
        let r = r0 + (r_out - r0) * k as f64 / lp as f64;
        [r * th(j % ns).cos(), r * th(j % ns).sin()]
    };
    for k in 0..lp {
        for j in 0..ns {
            out.push(quad(ring(k, j), ring(k + 1, j), ring(k + 1, j + 1), ring(k, j + 1)));
        }
    }
    // the blended layers between the square and the circle r0
    let lt = ((r0 - b) / s).ceil().max(1.0) as usize;
    let layer = |k: usize, j: usize| {
        let j = j % ns;
        let f = foot(j);
        let c = [r0 * th(j).cos(), r0 * th(j).sin()];
        let t = k as f64 / lt as f64;
        [f[0] + t * (c[0] - f[0]), f[1] + t * (c[1] - f[1])]
    };
    for k in 0..lt {
        for j in 0..ns {
            let p = quad(layer(k, j), layer(k + 1, j), layer(k + 1, j + 1), layer(k, j + 1));
            // at the square's corners the layers meet the circle: a triangle, or nothing
            let a = area(&p);
            if a > eps * eps {
                out.push(p);
            }
        }
    }
    // the outermost row of the square: the grid's last line (half-width b − h) against the feet
    let g = b - h;
    let m = n - 1; // grid steps from the centre to the last line
    let mut inner: Vec<[f64; 2]> = Vec::new();
    for k in 0..m {
        inner.push([g, h * k as f64]); // right side, up from the axis
    }
    for k in 0..2 * m {
        inner.push([g - h * k as f64, g]); // top
    }
    for k in 0..2 * m {
        inner.push([-g, g - h * k as f64]); // left
    }
    for k in 0..2 * m {
        inner.push([-g + h * k as f64, -g]); // bottom
    }
    for k in 0..m {
        inner.push([g, -g + h * k as f64]); // right side, up to the axis
    }
    let outer: Vec<[f64; 2]> = (0..ns).map(foot).collect();
    let ang = |x: [f64; 2]| x[1].atan2(x[0]).rem_euclid(2.0 * PI);
    let (ni, no) = (inner.len(), outer.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < ni || j < no {
        let ai = if i < ni { if i + 1 < ni { ang(inner[i + 1]) } else { 2.0 * PI } } else { f64::INFINITY };
        let aj = if j < no { if j + 1 < no { ang(outer[j + 1]) } else { 2.0 * PI } } else { f64::INFINITY };
        let (pi_, pj) = (inner[i % ni], outer[j % no]);
        if ai <= aj {
            out.push([pi_, pj, inner[(i + 1) % ni], inner[(i + 1) % ni]]);
            i += 1;
        } else {
            out.push([pi_, pj, outer[(j + 1) % no], outer[(j + 1) % no]]);
            j += 1;
        }
    }
    // the grid inside the last line: 2m x 2m cells; the cells clear of the waterlines are panels
    let c = 2 * m;
    let node = |i: usize, j: usize| [-g + h * i as f64, -g + h * j as f64];
    let mut keep = vec![false; c * c];
    for i in 0..c {
        for j in 0..c {
            let cs = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
            let clear = wl.iter().all(|lp| {
                cs.iter().all(|x| !inside(lp, *x))
                    && (0..lp.len()).all(|e| {
                        let (p, q) = (lp[e], lp[(e + 1) % lp.len()]);
                        !inside(&cs, p) && (0..4).all(|k| !meet(cs[k], cs[(k + 1) % 4], p, q))
                    })
            });
            keep[i * c + j] = clear;
        }
    }
    // a kept cell is one reached from the edge of the grid through kept cells (no islands)
    let mut reach = vec![false; c * c];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for i in 0..c {
        for j in 0..c {
            if (i == 0 || j == 0 || i == c - 1 || j == c - 1) && keep[i * c + j] {
                reach[i * c + j] = true;
                stack.push((i, j));
            }
        }
    }
    while let Some((i, j)) = stack.pop() {
        let mut nb = Vec::with_capacity(4);
        if i > 0 {
            nb.push((i - 1, j));
        }
        if j > 0 {
            nb.push((i, j - 1));
        }
        if i + 1 < c {
            nb.push((i + 1, j));
        }
        if j + 1 < c {
            nb.push((i, j + 1));
        }
        for (a, bb) in nb {
            if keep[a * c + bb] && !reach[a * c + bb] {
                reach[a * c + bb] = true;
                stack.push((a, bb));
            }
        }
    }
    for i in 0..c {
        for j in 0..c {
            if reach[i * c + j] {
                out.push(quad(node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)));
            }
        }
    }
    // the rest: the boundary edges of the cells not kept (the kept region or the last line on
    // their other side), counter-clockwise about the region, chained into loops
    let gone = |i: i64, j: i64| {
        i >= 0 && j >= 0 && (i as usize) < c && (j as usize) < c && !reach[i as usize * c + j as usize]
    };
    let mut edges: Vec<((usize, usize), (usize, usize))> = Vec::new();
    for i in 0..c as i64 {
        for j in 0..c as i64 {
            if !gone(i, j) {
                continue;
            }
            let (iu, ju) = (i as usize, j as usize);
            if !gone(i, j - 1) {
                edges.push(((iu, ju), (iu + 1, ju))); // bottom, left to right
            }
            if !gone(i + 1, j) {
                edges.push(((iu + 1, ju), (iu + 1, ju + 1))); // right, upwards
            }
            if !gone(i, j + 1) {
                edges.push(((iu + 1, ju + 1), (iu, ju + 1))); // top, right to left
            }
            if !gone(i - 1, j) {
                edges.push(((iu, ju + 1), (iu, ju))); // left, downwards
            }
        }
    }
    let mut used = vec![false; edges.len()];
    let mut regions: Vec<Vec<[f64; 2]>> = Vec::new();
    for e0 in 0..edges.len() {
        if used[e0] {
            continue;
        }
        used[e0] = true;
        let start = edges[e0].0;
        let mut lp = vec![node(start.0, start.1)];
        let mut cur = edges[e0];
        while cur.1 != start {
            // the next edge from the end; at a pinch point (two choices) the one turning right
            let from = cur.1;
            let cands: Vec<usize> = (0..edges.len()).filter(|&k| !used[k] && edges[k].0 == from).collect();
            let Some(&k) = cands.iter().min_by(|&&x, &&y| {
                let dir = |e: ((usize, usize), (usize, usize))| {
                    let d0 = [cur.1.0 as f64 - cur.0.0 as f64, cur.1.1 as f64 - cur.0.1 as f64];
                    let d1 = [e.1.0 as f64 - e.0.0 as f64, e.1.1 as f64 - e.0.1 as f64];
                    d0[0] * d1[1] - d0[1] * d1[0] // > 0: a left turn
                };
                dir(edges[x]).partial_cmp(&dir(edges[y])).unwrap()
            }) else {
                return Err(Error::internal("free-surface panels: an open boundary of the waterline cells"));
            };
            used[k] = true;
            lp.push(node(from.0, from.1));
            cur = edges[k];
        }
        regions.push(lp);
    }
    for outer in regions {
        if area(&outer) <= 0.0 {
            return Err(Error::internal("free-surface panels: a waterline region turned the wrong way"));
        }
        let holes: Vec<Vec<[f64; 2]>> = wl
            .iter()
            .filter(|lp| inside(&outer, lp[0]))
            .map(|lp| {
                let mut h = lp.clone();
                h.reverse();
                h
            })
            .collect();
        out.extend(triangulate(&Region { outer, holes }, eps)?.into_iter().map(|[a, b, c]| [a, b, c, c]));
    }
    Ok(out.into_iter().map(|p| p.map(|x| [x[0] + c0[0], x[1] + c0[1]])).collect())
}

/// The centroid and the area of a panel (a triangle with its third corner twice).
pub fn centroid_area(p: &Panel) -> ([f64; 2], f64) {
    let a = area(p);
    let (mut cx, mut cy) = (0.0, 0.0);
    for k in 0..4 {
        let (x, y) = (p[k], p[(k + 1) % 4]);
        let w = x[0] * y[1] - y[0] * x[1];
        cx += (x[0] + y[0]) * w;
        cy += (x[1] + y[1]) * w;
    }
    ([cx / (6.0 * a), cy / (6.0 * a)], a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polygon(n: usize, r: f64) -> Vec<([f64; 2], [f64; 2])> {
        (0..n)
            .map(|k| {
                let (a, b) = (2.0 * PI * k as f64 / n as f64, 2.0 * PI * (k + 1) as f64 / n as f64);
                ([r * a.cos(), r * a.sin()], [r * b.cos(), r * b.sin()])
            })
            .collect()
    }

    /// The area of the polygon of the outer circle's nodes (the panels on it counted).
    fn outer(p: &[Panel], r: f64) -> f64 {
        let ns =
            p.iter().filter(|x| x[..3].iter().filter(|v| (v[0].hypot(v[1]) - r).abs() < 1e-9).count() >= 2).count();
        0.5 * ns as f64 * (2.0 * PI / ns as f64).sin() * r * r
    }

    #[test]
    fn the_panels_tile_the_free_surface() {
        // a 48-gon of radius 1 (12 sides a quadrant), R = 4: the areas sum to the outer polygon's
        // less the waterline's; every panel counter-clockwise, its centroid outside the waterline;
        // the rings and spokes as the rule gives them (14 x 30 a quadrant for s = 0.15)
        for (n, s, rings) in [(48usize, 0.15, 14 * 120), (96, 0.15, 14 * 120), (96, 0.075, 27 * 240)] {
            let segs = polygon(n, 1.0);
            let p = panels(&segs, [0.0, 0.0], 4.0, s).unwrap();
            let poly = 0.5 * n as f64 * (2.0 * PI / n as f64).sin();
            let total: f64 = p.iter().map(|x| centroid_area(x).1).sum();
            assert!((total - (outer(&p, 4.0) - poly)).abs() <= 1e-9, "{n} {s}: {total}");
            let wl: Vec<[f64; 2]> = segs.iter().map(|x| x.0).collect();
            for x in &p {
                assert!(area(x) > 0.0);
                assert!(!inside(&wl, centroid_area(x).0), "{n} {s} {x:?}");
            }
            let r0 = 2.0 * (PI / n as f64).cos();
            let polar = p.iter().filter(|x| x.iter().all(|v| v[0].hypot(v[1]) >= r0 - 1e-9)).count();
            assert_eq!(polar, rings, "{n} {s}");
        }
        // two bodies apart, about their centre
        let mut segs = polygon(24, 0.5);
        for s in &mut segs {
            s.0[0] += 1.5;
            s.1[0] += 1.5;
        }
        let mut two = polygon(24, 0.5);
        for s in &mut two {
            s.0[0] -= 1.5;
            s.1[0] -= 1.5;
        }
        segs.extend(two);
        let p = panels(&segs, [0.0, 0.0], 6.0, 0.2).unwrap();
        let total: f64 = p.iter().map(|x| centroid_area(x).1).sum();
        let poly = 2.0 * 0.5 * 24.0 * 0.25 * (2.0 * PI / 24.0).sin();
        assert!((total - (outer(&p, 6.0) - poly)).abs() <= 1e-9, "{total}");
    }

    #[test]
    fn a_coarse_waterline_stays_inside_the_grid() {
        // a 12-gon of radius 1 with panels of 0.5: by the size alone 3 steps, the grid's last line
        // at 0.91 inside the corners at 1 (the waterline would cross the outer row, the panels
        // overlap the body); the waterline inside it takes 5. The panels tile the free surface
        // (the areas sum to the outer polygon's less the 12-gon's), none has a corner inside the
        // waterline. A triangle (its corners at twice its sides' middles, beyond the square) is
        // refused.
        let segs = polygon(12, 1.0);
        let p = panels(&segs, [0.0, 0.0], 4.0, 0.5).unwrap();
        let poly = 0.5 * 12.0 * (2.0 * PI / 12.0).sin();
        let total: f64 = p.iter().map(|x| centroid_area(x).1).sum();
        assert!((total - (outer(&p, 4.0) - poly)).abs() <= 1e-9, "{total}");
        let wl: Vec<[f64; 2]> = segs.iter().map(|x| x.0).collect();
        let strictly = |x: &[f64; 2]| {
            inside(&wl, *x) && (0..12).all(|k| crate::polygon::orient(wl[k], wl[(k + 1) % 12], *x) > 1e-9)
        };
        assert!(p.iter().all(|x| !x.iter().any(strictly)));
        let e = panels(&polygon(3, 1.0), [0.0, 0.0], 4.0, 0.2).unwrap_err();
        assert!(e.message().contains("too coarse for the generated panels"), "{}", e.message());
    }

    #[test]
    fn a_waterline_inside_another_is_refused() {
        // an annulus (a moonpool of radius 0.5 in a body of radius 1): refused with a value error,
        // not triangulated over the waterplane
        let mut segs = polygon(24, 1.0);
        segs.extend(polygon(12, 0.5));
        let e = panels(&segs, [0.0, 0.0], 4.0, 0.2).unwrap_err();
        assert_eq!(e.kind(), crate::error::ErrorKind::Value);
        assert!(e.message().contains("a waterline lies inside another (a moonpool)"), "{}", e.message());
    }
}
