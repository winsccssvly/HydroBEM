//! Thin sheets of the higher-order method: the Rankine and wave parts of the rows of the
//! planar dipole patches.

use crate::bspline::basis;
use crate::geometry::{Vec3, cross, dot, norm, v3};
use crate::green_fin::{FinPoint, FreqMode, fin_point};
use crate::ho_geometry::point_modes;
use crate::ho_internal::{
    BlockRule, HoGeom, Inst, MAX_MODES, SheetPt, SheetRowData, WAVE_FAR, inst_eval, inst_tangents,
};
use crate::low_order::{FreqCtx, SurfaceKind, channel_far_sheet, sheet_wave, sheet_wave_fd};
use crate::panel_integrals::{PanelGeom, make_panel_geom, rankine_grad};
use crate::quadrature::gauss_legendre;
use num_complex::Complex64;

// The Rankine terms of the kernel (sign, image point and image normal of the source y):
// infinite depth y and its free-surface image (+ at omega = 0 and ordinary, - at
// omega = inf); finite depth the six terms of green_fin.rs with fd.sign. The images of ho_static.rs
// in the same order, there of the target x (the source image y - 2h is the target image x + 2h);
// here the source is imaged, since the normal of the image source enters the second derivative.
#[inline]
fn rankine_images(
    depth: f64,
    inf: bool,
    y: &Vec3,
    ny: &Vec3,
    s: &mut [f64; 6],
    yp: &mut [Vec3; 6],
    np: &mut [Vec3; 6],
) -> usize {
    let flip = v3(ny[0], ny[1], -ny[2]);
    if !depth.is_finite() {
        s[0] = 1.0;
        yp[0] = *y;
        np[0] = *ny;
        s[1] = if inf { -1.0 } else { 1.0 };
        yp[1] = v3(y[0], y[1], -y[2]);
        np[1] = flip;
        return 2;
    }
    let h = depth;
    let z = y[2];
    // v1 = |z - zeta|, v2 = z + zeta + 2h, v3 = -(z + zeta), v4 = z - zeta + 2h, v5 = zeta - z + 2h,
    // v6 = z + zeta + 4h as distances to the images zeta_p = zeta, -zeta - 2h, -zeta, zeta - 2h,
    // zeta + 2h, -zeta - 4h (the reflections flip the normal)
    let zp = [z, -z - 2.0 * h, -z, z - 2.0 * h, z + 2.0 * h, -z - 4.0 * h];
    let fl = [false, true, true, false, false, true];
    for p in 0..6 {
        s[p] = if p >= 2 && inf { -1.0 } else { 1.0 }; // green_fin: (+, +, +, +, +, +), at omega = inf (+, +, -, -, -, -)
        yp[p] = v3(y[0], y[1], zp[p]);
        np[p] = if fl[p] { flip } else { *ny };
    }
    6
}

// Rankine 1/r: the second normal derivative and the field normal derivative (H, Kp) at x of a
// unit source at y.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn rankine_pair(x: &Vec3, nx: &Vec3, y: &Vec3, ny: &Vec3) -> (f64, f64) {
    let R = *x - *y;
    let r2 = dot(R, R);
    let r = r2.sqrt();
    let ir3 = 1.0 / (r2 * r);
    let ir5 = ir3 / r2;
    (dot(*nx, *ny) * ir3 - 3.0 * dot(*nx, R) * dot(*ny, R) * ir5, -dot(*nx, R) * ir3)
}

// The index of the free-surface image in rankine_images.
#[inline]
fn free_surface_image(depth: f64) -> usize {
    if depth.is_finite() { 2 } else { 1 }
}

// The Rankine terms of the kernel at (x, nx) of a unit source/dipole at (y, ny): (H, Kp);
// `self_` omits the direct term (handled by the singular subtraction on the own patch), `no_fs`
// the free-surface image too (a plate's own patch at omega = inf, where the image coincides with
// the direct term and is counted by doubling it).
#[inline]
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn sheet_rankine(
    depth: f64,
    inf: bool,
    x: &Vec3,
    nx: &Vec3,
    y: &Vec3,
    ny: &Vec3,
    self_: bool,
    no_fs: bool,
) -> (f64, f64) {
    let mut s = [0.0f64; 6];
    let mut yp = [Vec3::ZERO; 6];
    let mut np = [Vec3::ZERO; 6];
    let n = rankine_images(depth, inf, y, ny, &mut s, &mut yp, &mut np);
    let (mut hr, mut kr) = (0.0, 0.0);
    let fs = free_surface_image(depth);
    for p in (if self_ { 1 } else { 0 })..n {
        if no_fs && p == fs {
            continue;
        }
        let (a, b) = rankine_pair(x, nx, &yp[p], &np[p]);
        hr += s[p] * a;
        kr += s[p] * b;
    }
    (hr, kr)
}

// The wave (regular) part (H, Kp) at (x, nx) of a unit source/dipole at (y, ny).
#[inline]
fn sheet_wave_point(
    fc: &FreqCtx,
    x: &Vec3,
    nx: &Vec3,
    y: &Vec3,
    ny: &Vec3,
    px: Option<&FinPoint>,
    py: Option<&FinPoint>,
) -> (Complex64, Complex64) {
    let pg = PanelGeom { c: *y, n: *ny, area: 1.0, ..PanelGeom::default() };
    if fc.finite() {
        sheet_wave_fd(fc, x, nx, &pg, px, py)
    } else if fc.mode == FreqMode::Ordinary {
        sheet_wave(fc, x, nx, &pg)
    } else {
        (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0))
    }
}

// Gauss points on [0, 1].
fn gauss01(n: i32) -> (Vec<f64>, Vec<f64>) {
    let (mut t, mut w) = gauss_legendre(n);
    for i in 0..n as usize {
        t[i] = 0.5 * (t[i] + 1.0);
        w[i] *= 0.5;
    }
    (t, w)
}

// the Gauss rules on [0, 1] of the cells (6), the line integrals (16) and the apex cell (10),
// computed once per thread
thread_local! {
    static G6: (Vec<f64>, Vec<f64>) = gauss01(6);
    static G10: (Vec<f64>, Vec<f64>) = gauss01(10);
    static G16: (Vec<f64>, Vec<f64>) = gauss01(16);
}

// Integrate over a parametric cell [u0, u1] x [v0, v1] of instance ii with the
// callback f(u, v, x, n, J dudv), subdividing while the cell is large against its
// distance to the target points (the field point and its images).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn sheet_cell<F: FnMut(f64, f64, &Vec3, &Vec3, f64)>(
    h: &HoGeom,
    ii: i32,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
    tg: &[Vec3],
    level: i32,
    f: &mut F,
) {
    let in_ = &h.inst[ii as usize];
    let (mut xc, mut nc) = (Vec3::ZERO, Vec3::ZERO);
    let mut J = 0.0;
    inst_eval(h, in_, 0.5 * (u0 + u1), 0.5 * (v0 + v1), &mut xc, &mut nc, &mut J, None, None);
    let mut size = 0.0f64;
    for a in 0..2 {
        for b in 0..2 {
            let (mut xk, mut nk) = (Vec3::ZERO, Vec3::ZERO);
            let mut Jk = 0.0;
            inst_eval(
                h,
                in_,
                if a != 0 { u1 } else { u0 },
                if b != 0 { v1 } else { v0 },
                &mut xk,
                &mut nk,
                &mut Jk,
                None,
                None,
            );
            size = size.max(norm(xk - xc));
        }
    }
    let mut d = f64::INFINITY;
    for &t in tg {
        d = d.min(norm(t - xc));
    }
    // a cell nearer to a target than 2.5 of its sizes is quadrisected: beyond, the 6 x 6 Gauss rule
    // errs by about rho^-12 of the integrand, rho = 2.5 + (2.5^2 - 1)^(1/2) = 4.8 (the ellipse of
    // analyticity through the nearest target), 7e-9. The criterion ends the subdivision for a
    // target off the cell after about log2(2.5 size / distance) levels: the distance of a Gauss point
    // from a side goes as the square of its parameter where the profile has zero slope, so it needs
    // more than a fixed few (at a disc's free edge 7 levels left 9e-4 of the added mass at 12
    // spans); 40 levels (2^-40 of the span, the rounding of the parameters) bound it for a target on
    // the cell's boundary
    if level < 40 && d < 2.5 * size {
        let um = 0.5 * (u0 + u1);
        let vm = 0.5 * (v0 + v1);
        sheet_cell(h, ii, u0, um, v0, vm, tg, level + 1, f);
        sheet_cell(h, ii, um, u1, v0, vm, tg, level + 1, f);
        sheet_cell(h, ii, u0, um, vm, v1, tg, level + 1, f);
        sheet_cell(h, ii, um, u1, vm, v1, tg, level + 1, f);
        return;
    }
    G6.with(|(gt, gw)| {
        for i in 0..6 {
            for j in 0..6 {
                let u = u0 + (u1 - u0) * gt[i];
                let v = v0 + (v1 - v0) * gt[j];
                let (mut x, mut n) = (Vec3::ZERO, Vec3::ZERO);
                let mut Jp = 0.0;
                inst_eval(h, in_, u, v, &mut x, &mut n, &mut Jp, None, None);
                f(u, v, &x, &n, Jp * (u1 - u0) * (v1 - v0) * gw[i] * gw[j]);
            }
        }
    });
}

// The singular terms of a planar patch with curved sides by line integrals over its boundary:
// fp = FP int 1/r^3 dS and lin = grad_x int 1/r dS (in-plane) at x in the patch.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn planar_terms(h: &HoGeom, in0: &Inst, x: &Vec3, xu0: &Vec3, xv0: &Vec3, fp: &mut f64, lin: &mut Vec3) {
    G16.with(|(gt, gw)| {
        *fp = 0.0;
        *lin = v3(0.0, 0.0, 0.0);
        // the four sides counter-clockwise about x_u x x_v, each in pieces fine near x
        let su: [[f64; 2]; 4] = [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];
        let sv: [[f64; 2]; 4] = [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]];
        let nl = (1.0 / norm(cross(*xu0, *xv0))) * cross(*xu0, *xv0);
        for e in 0..4 {
            // pieces by bisection: a piece is split while longer than half the smallest distance from x
            // to its 16 nodes and its two ends. The distance to the piece is then at least about twice
            // its length, where the 16-point rule errs by about (4 + 15^(1/2))^-32, below rounding; a
            // point a distance delta from the side gets about log2(length / delta) levels next to its
            // foot point and coarse pieces elsewhere. The distance must be that of the piece itself: one
            // sampling of the whole side misses a point close to the side between its samples (pieces
            // then far longer than the distance, the peak of R.m / r^3 of width about delta missed).
            // The length is the Gauss sum of |dy/ds| (a closed side, a full circle, has no chord).
            let side = |s: f64, y: &mut Vec3, t: &mut Vec3| {
                let u = su[e][0] + s * (su[e][1] - su[e][0]);
                let v = sv[e][0] + s * (sv[e][1] - sv[e][0]);
                let (mut ny, mut yu, mut yv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                let mut Jy = 0.0;
                inst_eval(h, in0, u, v, y, &mut ny, &mut Jy, None, None);
                inst_tangents(h, in0, u, v, &mut yu, &mut yv);
                *t = (su[e][1] - su[e][0]) * yu + (sv[e][1] - sv[e][0]) * yv; // dy/ds
            };
            let mut stack: Vec<(f64, f64, u32)> = vec![(0.0, 1.0, 0)];
            while let Some((a, b, level)) = stack.pop() {
                let mut ys = [Vec3::ZERO; 18];
                let mut ts = [Vec3::ZERO; 18];
                for i in 0..16 {
                    side(a + (b - a) * gt[i], &mut ys[i], &mut ts[i]);
                }
                side(a, &mut ys[16], &mut ts[16]);
                side(b, &mut ys[17], &mut ts[17]);
                let dmin = ys.iter().fold(f64::INFINITY, |d, y| d.min(norm(*y - *x)));
                let len: f64 = (0..16).map(|i| gw[i] * (b - a) * norm(ts[i])).sum();
                // 48 levels: a piece of 2^-48 of the side, far below any distance of a Gauss point
                if level < 48 && len > 0.5 * dmin {
                    let mid = 0.5 * (a + b);
                    stack.push((a, mid, level + 1));
                    stack.push((mid, b, level + 1));
                    continue;
                }
                for i in 0..16 {
                    let t = ts[i];
                    let dl = norm(t);
                    if !(dl > 0.0) {
                        continue; // a degenerate side (the axis of a disc)
                    }
                    let m = (1.0 / dl) * cross(t, nl);
                    let R = ys[i] - *x;
                    let r = norm(R);
                    let w = gw[i] * (b - a) * dl;
                    *fp -= w * dot(R, m) / (r * r * r);
                    *lin -= (w / r) * m;
                }
            }
        }
    });
}

/// The wave part of the sheet integrals at sp, added to out (per frequency); fin_in, fin_far:
/// the finite-depth one-point factors of the inner and far points (empty in infinite depth).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn sheet_wave_row(
    h: &HoGeom,
    fc: &FreqCtx,
    sp: &SheetPt,
    fin_in: &[FinPoint],
    fin_far: &[FinPoint],
    out: &mut SheetRowData,
) {
    let NM = h.NM as usize;
    let x = &sp.x;
    let nx = &sp.n;
    // the wave part: the inner points of every (instance, panel) block (the lower-order rule
    // far from x and its images), as kernel_row
    if fc.finite() || fc.mode == FreqMode::Ordinary {
        let fx = if fc.finite() { fin_point(&fc.fd, x[2]) } else { FinPoint::default() };
        let nbk = h.inner0.len() - 1;
        for bk in 0..nbk {
            let a0 = h.inner0[bk] as usize;
            let e0 = h.inner0[bk + 1] as usize;
            if a0 == e0 {
                continue;
            }
            let ii = h.inner[a0].inst;
            let kd = h.kind_of(ii);
            let nk = h.bnk[bk] as usize;
            let c = &h.bc[bk];
            let rx = x[0] - c[0];
            let ry = x[1] - c[1];
            let mut rz = (x[2] + c[2]).abs();
            if h.depth.is_finite() {
                rz = rz.min((x[2] + c[2] + 2.0 * h.depth).abs());
            }
            let far = (rx * rx + ry * ry + rz * rz) > WAVE_FAR * WAVE_FAR * h.bs[bk] * h.bs[bk];
            let mut wo = if far { h.f0[bk] as usize * nk } else { h.bwb0[bk] as usize }; // offset of the weights
            let wbs: &[f64] = if far { &h.fwb } else { &h.wb };
            let q0 = if far { h.f0[bk] as usize } else { a0 };
            let q1 = if far { h.f0[bk + 1] as usize } else { e0 };
            let X: &[f64] = if far { &h.fx } else { &h.wx };
            let Nn: &[f64] = if far { &h.fn_ } else { &h.wn };
            let Wm: &[f64] = if far { &h.fm } else { &h.wm };
            let col = &h.bcol[h.bcol0[bk] as usize..];
            for qi in q0..q1 {
                let wb = &wbs[wo..];
                wo += nk;
                let y = v3(X[3 * qi], X[3 * qi + 1], X[3 * qi + 2]);
                let ny = v3(Nn[3 * qi], Nn[3 * qi + 1], Nn[3 * qi + 2]);
                let (Hw, Kw) = if fc.finite() {
                    sheet_wave_point(fc, x, nx, &y, &ny, Some(&fx), Some(if far { &fin_far[qi] } else { &fin_in[qi] }))
                } else {
                    sheet_wave_point(fc, x, nx, &y, &ny, None, None)
                };
                for t in 0..nk {
                    out.H[col[t] as usize] += wb[t] * Hw;
                    if kd != SurfaceKind::Sheet {
                        out.Kc[col[t] as usize] += wb[t] * Kw;
                    }
                }
                if kd != SurfaceKind::Sheet {
                    let wm = &Wm[qi * NM..];
                    for k in 0..NM {
                        if wm[k] != 0.0 {
                            out.Mn[ii as usize * NM + k] += wm[k] * Kw;
                        }
                    }
                }
            }
        }
    }
    if h.W > 0.0 {
        channel_sheet_terms(h, fc, sp, fin_in, fin_far, out);
    }
}

/// The channel terms of a sheet row, those of the wave rows (channel_terms): the columns of the
/// wall images (Rankine and wave parts) scaled by the reflection factor, and the farther images
/// of every original instance (channel_far_sheet) at its points: the far rule, or the inner rule
/// within WAVE_FAR sizes of the nearest far image.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn channel_sheet_terms(
    h: &HoGeom,
    fc: &FreqCtx,
    sp: &SheetPt,
    fin_in: &[FinPoint],
    fin_far: &[FinPoint],
    out: &mut SheetRowData,
) {
    let NM = h.NM as usize;
    let Rc = h.scale_wall_images(&mut out.H, Some(&mut out.Kc), &mut out.Mn);
    if Rc == 0.0 {
        return;
    }
    let (W, x, nx) = (h.W, &sp.x, &sp.n);
    let fx = fin_point(&fc.fd, x[2]);
    for bk in 0..h.inner0.len() - 1 {
        let a0 = h.inner0[bk] as usize;
        let e0 = h.inner0[bk + 1] as usize;
        if a0 == e0 || h.inst[h.inner[a0].inst as usize].image {
            continue;
        }
        let ii = h.inner[a0].inst as usize;
        let sheet = h.kind_of(ii as i32) == SurfaceKind::Sheet;
        let nk = h.bnk[bk] as usize;
        let far = h.far_from_wall_images(bk, x);
        let BlockRule { q0, q1, wb: wbv, wb0: mut wbo, x: X, n: Nn, m: Wm } = h.block_rule(bk, far);
        let fq = if far { fin_far } else { fin_in };
        let col = &h.bcol[h.bcol0[bk] as usize..];
        for qi in q0..q1 {
            let y = v3(X[3 * qi], X[3 * qi + 1], X[3 * qi + 2]);
            let ny = v3(Nn[3 * qi], Nn[3 * qi + 1], Nn[3 * qi + 2]);
            let (Hf, Kf) = channel_far_sheet(fc, W, x, nx, &y, &ny, Rc, &fx, &fq[qi]);
            let wb = &wbv[wbo..wbo + nk];
            for t in 0..nk {
                out.H[col[t] as usize] += wb[t] * Hf;
                if !sheet {
                    out.Kc[col[t] as usize] += wb[t] * Kf;
                }
            }
            if !sheet {
                let wm = &Wm[qi * NM..qi * NM + NM];
                for k in 0..NM {
                    out.Mn[ii * NM + k] += wm[k] * Kf;
                }
            }
            wbo += nk;
        }
    }
}

/// The Rankine part of the sheet integrals at sp (frequency independent; inf: the signs of
/// omega = inf): out is assigned.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn sheet_rankine_row(h: &HoGeom, depth: f64, inf: bool, sp: &SheetPt, out: &mut SheetRowData) {
    let NM = h.NM as usize;
    out.H = vec![Complex64::new(0.0, 0.0); h.Ncol as usize];
    out.Kc = vec![Complex64::new(0.0, 0.0); h.Ncol as usize];
    out.Mn = vec![Complex64::new(0.0, 0.0); h.inst.len() * NM];
    let x = sp.x;
    let nx = sp.n;
    // targets of the subdivision: x and its images in the free surface (and the bottom)
    let tg = [x, v3(x[0], x[1], -x[2]), v3(x[0], x[1], -x[2] - 2.0 * depth)];
    let ntg = if depth.is_finite() { 3 } else { 2 };
    // the own patch: B(x) and the surface gradient of the basis at x
    let in0 = &h.inst[sp.inst as usize];
    let d0 = &h.disc[in0.patch as usize];
    let (mut Nu0, mut dNu0, mut Nv0, mut dNv0) = ([0.0f64; 8], [0.0f64; 8], [0.0f64; 8], [0.0f64; 8]);
    basis(&d0.Tu, d0.ku, sp.iu + d0.ku - 1, sp.u, &mut Nu0, Some(&mut dNu0));
    basis(&d0.Tv, d0.kv, sp.iv + d0.kv - 1, sp.v, &mut Nv0, Some(&mut dNv0));
    let (mut xu0, mut xv0) = (Vec3::ZERO, Vec3::ZERO);
    inst_tangents(h, in0, sp.u, sp.v, &mut xu0, &mut xv0);
    let ga = dot(xu0, xu0);
    let gb = dot(xu0, xv0);
    let gc = dot(xv0, xv0);
    let det = ga * gc - gb * gb;
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    struct Act {
        col: i32,
        B: f64,
        g: Vec3,
    }
    // a free-surface plate at omega = inf (x and its own patch in z = 0): the odd free-surface
    // image of the own patch coincides with it and doubles its singular terms (theory §9.8), so
    // the closed forms and the subtracted direct term count twice and that image is left out
    let dbl = inf && h.kind[in0.patch as usize] == SurfaceKind::Plate;
    let fac = if dbl { 2.0 } else { 1.0 };
    let mut act: Vec<Act> = Vec::new();
    let mu0 = d0.mu();
    for b in 0..d0.kv as usize {
        for a in 0..d0.ku as usize {
            let fu = dNu0[a] * Nv0[b];
            let fv = Nu0[a] * dNv0[b];
            let p = (gc * fu - gb * fv) / det;
            let q = (ga * fv - gb * fu) / det;
            act.push(Act {
                col: h.col0[sp.inst as usize] + (sp.iv + b as i32) * mu0 + sp.iu + a as i32,
                B: Nu0[a] * Nv0[b],
                g: p * xu0 + q * xv0,
            });
        }
    }
    // the closed forms on the own patch: FP int 1/r^3 dS = n . grad_x (dipole) and
    // int (xi - x) / r^3 dS = grad_x int 1/r dS (in-plane part); a flat patch by its polygon,
    // a planar patch with curved sides by the line integrals of the divergence theorem
    // in its plane, FP int 1/r^3 dS = -oint (y - x) . m / r^3 dl and grad int 1/r dS = -oint m / r dl,
    // m the outward normal of the boundary in the plane
    if !h.geo[in0.patch as usize].flat {
        let mut fp = 0.0;
        let mut lin = Vec3::ZERO;
        planar_terms(h, in0, &x, &xu0, &xv0, &mut fp, &mut lin);
        for a in &act {
            out.H[a.col as usize] += fac * (a.B * fp + dot(a.g, lin));
        }
    } else {
        let mut v4 = [Vec3::ZERO; 4];
        for k in 0..4 {
            let uu = if k == 1 || k == 2 { 1.0 } else { -1.0 };
            let vv = if k >= 2 { 1.0 } else { -1.0 };
            let (mut xk, mut nk) = (Vec3::ZERO, Vec3::ZERO);
            let mut Jk = 0.0;
            inst_eval(h, in0, uu, vv, &mut xk, &mut nk, &mut Jk, None, None);
            v4[k] = xk;
        }
        if dot(cross(v4[1] - v4[0], v4[3] - v4[0]), nx) < 0.0 {
            v4.swap(1, 3);
        }
        let mut c = v3(0.0, 0.0, 0.0);
        let mut area = 0.0;
        for k in 1..3 {
            let a = 0.5 * norm(cross(v4[k] - v4[0], v4[k + 1] - v4[0]));
            c += (a / 3.0) * (v4[0] + v4[k] + v4[k + 1]);
            area += a;
        }
        c = (1.0 / area) * c;
        let pg = make_panel_geom(&v4, &nx, &c, area);
        let (gS, gD) = rankine_grad(&pg, &x);
        let fp = dot(nx, gD);
        let lin = gS - dot(gS, nx) * nx;
        for a in &act {
            out.H[a.col as usize] += fac * (a.B * fp + dot(a.g, lin));
        }
    }
    // every instance, the Rankine terms: cells of the potential panels, adaptive Gauss; the own
    // patch with the subtraction B(xi) - B(x) - grad B(x) . (xi - x) in its direct term
    for ii in 0..h.inst.len() {
        let in_ = &h.inst[ii];
        let kd = h.kind[in_.patch as usize];
        let own = ii as i32 == sp.inst;
        let d = &h.disc[in_.patch as usize];
        let mu = d.mu();
        let c0 = h.col0[ii];
        for iv in 0..d.nv {
            for iu in 0..d.nu {
                let u0 = -1.0 + 2.0 * iu as f64 / d.nu as f64;
                let u1 = -1.0 + 2.0 * (iu + 1) as f64 / d.nu as f64;
                let v0 = -1.0 + 2.0 * iv as f64 / d.nv as f64;
                let v1 = -1.0 + 2.0 * (iv + 1) as f64 / d.nv as f64;
                let mut point = |u: f64, v: f64, y: &Vec3, ny: &Vec3, w: f64| {
                    // w: the weight of the regular terms and of the subtracted direct term
                    let (mut Nu, mut dNu, mut Nv, mut dNv) = ([0.0f64; 8], [0.0f64; 8], [0.0f64; 8], [0.0f64; 8]);
                    basis(&d.Tu, d.ku, iu + d.ku - 1, u, &mut Nu, Some(&mut dNu));
                    basis(&d.Tv, d.kv, iv + d.kv - 1, v, &mut Nv, Some(&mut dNv));
                    let (Hk, Kk) = sheet_rankine(depth, inf, &x, &nx, y, ny, own, own && dbl);
                    let H0 = if own { fac * rankine_pair(&x, &nx, y, ny).0 } else { 0.0 };
                    for b in 0..d.kv {
                        for a in 0..d.ku {
                            let col = (c0 + (iv + b) * mu + iu + a) as usize;
                            let B = Nu[a as usize] * Nv[b as usize];
                            out.H[col] += w * B * Hk + w * B * H0;
                            if kd != SurfaceKind::Sheet {
                                out.Kc[col] += w * B * Kk;
                            }
                        }
                    }
                    if own {
                        for a in &act {
                            out.H[a.col as usize] -= w * (a.B + dot(a.g, *y - x)) * H0;
                        }
                    }
                    if kd != SurfaceKind::Sheet {
                        let (mut xb, mut nb, mut xx, mut nn) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                        let mut J = 0.0;
                        inst_eval(h, in_, u, v, &mut xx, &mut nn, &mut J, Some(&mut xb), Some(&mut nb));
                        let mut nm = [0.0f64; MAX_MODES];
                        point_modes(h, ii as i32, &xb, &nb, &mut nm);
                        for k in 0..NM {
                            if nm[k] != 0.0 {
                                out.Mn[ii * NM + k] += w * nm[k] * Kk;
                            }
                        }
                    }
                };
                // the collocation point in the cell to the rounding of the parameters (which span
                // [-1, 1]): 1e-12
                let apex = own && sp.u >= u0 - 1e-12 && sp.u <= u1 + 1e-12 && sp.v >= v0 - 1e-12 && sp.v <= v1 + 1e-12;
                if apex {
                    // the cell of the collocation point: four rectangles with their corner at it, each
                    // two Duffy triangles u = u_c + a p, v = v_c + b p q and v = v_c + b p,
                    // u = u_c + a p q, 10 x 10 points, graded in q as the body patches
                    // (duffy_rect of ho_static.rs): with rho the physical leg along q over the leg
                    // along p, for rho > 2 the q range is split at 1/rho, 2/rho, 4/rho, ... The
                    // kernel varies in q on the scale 1/rho near q = 0; a Gauss point next to a side
                    // of zero profile slope makes the rectangle towards that side flat (its legs
                    // 1e-4 and 0.1 of the patch), where one rule of 10 x 10 points left 1.4e-3 of a
                    // disc's added mass
                    G10.with(|(gt, gw)| {
                        let at = |u: f64, v: f64| {
                            let (mut y, mut ny) = (Vec3::ZERO, Vec3::ZERO);
                            let mut J = 0.0;
                            inst_eval(h, in_, u, v, &mut y, &mut ny, &mut J, None, None);
                            (y, ny, J)
                        };
                        let xc = at(sp.u, sp.v).0;
                        for a in [u0 - sp.u, u1 - sp.u] {
                            for b in [v0 - sp.v, v1 - sp.v] {
                                // a rectangle of no parametric extent (the point on that side of
                                // the cell; the rounding of the parameters is 1e-12, of a product
                                // of two 1e-16)
                                if a.abs() < 1e-14 || b.abs() < 1e-14 {
                                    continue;
                                }
                                let la = norm(at(sp.u + a, sp.v).0 - xc);
                                let lb = norm(at(sp.u, sp.v + b).0 - xc);
                                // a leg of no physical length (a collapsed side): no area
                                if !(la > 0.0 && lb > 0.0) {
                                    continue;
                                }
                                let ab = (a * b).abs();
                                for t in 0..2 {
                                    let rho = if t == 0 { lb / la } else { la / lb };
                                    let mut q0 = 0.0;
                                    while q0 < 1.0 {
                                        let q1 = if rho > 2.0 {
                                            1.0f64.min(if q0 == 0.0 { 1.0 / rho } else { 2.0 * q0 })
                                        } else {
                                            1.0
                                        };
                                        let hq = q1 - q0;
                                        for i in 0..10 {
                                            for j in 0..10 {
                                                let p = gt[i];
                                                let q = q0 + hq * gt[j];
                                                let (u, v) = if t == 0 {
                                                    (sp.u + a * p, sp.v + b * p * q)
                                                } else {
                                                    (sp.u + a * p * q, sp.v + b * p)
                                                };
                                                let (y, ny, J) = at(u, v);
                                                point(u, v, &y, &ny, J * ab * p * gw[i] * hq * gw[j]);
                                            }
                                        }
                                        q0 = q1;
                                    }
                                }
                            }
                        }
                    });
                } else {
                    sheet_cell(h, ii as i32, u0, u1, v0, v1, &tg[..ntg], 0, &mut |u, v, y: &Vec3, ny: &Vec3, w| {
                        point(u, v, y, ny, w)
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn rankine_images_and_pair() {
        let y = v3(0.5, -0.25, -1.0);
        let ny = v3(0.0, 0.6, 0.8);
        let (mut s, mut yp, mut np) = ([0.0; 6], [Vec3::ZERO; 6], [Vec3::ZERO; 6]);
        assert_eq!(rankine_images(f64::INFINITY, true, &y, &ny, &mut s, &mut yp, &mut np), 2);
        assert_eq!((s[0], s[1]), (1.0, -1.0));
        assert_eq!(yp[1], v3(0.5, -0.25, 1.0));
        assert_eq!(np[1], v3(0.0, 0.6, -0.8));
        assert_eq!(rankine_images(4.0, true, &y, &ny, &mut s, &mut yp, &mut np), 6);
        assert_eq!(s, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(yp[1][2], -7.0);
        assert_eq!(yp[5][2], -15.0);
        assert_eq!(np[3], ny);
        // H and Kp against central differences of 1/r
        let x = v3(1.0, 0.5, -0.4);
        let nx = v3(0.0, 0.0, 1.0);
        let (H, Kp) = rankine_pair(&x, &nx, &y, &ny);
        let g = |a: Vec3, b: Vec3| 1.0 / norm(a - b);
        let e = 1e-4;
        let kf = (g(x + e * nx, y) - g(x + (-e) * nx, y)) / (2.0 * e);
        let hf = (g(x + e * nx, y + e * ny) - g(x + e * nx, y + (-e) * ny) - g(x + (-e) * nx, y + e * ny)
            + g(x + (-e) * nx, y + (-e) * ny))
            / (4.0 * e * e);
        assert!((kf - Kp).abs() < 1e-6 * Kp.abs().max(1.0), "{kf} {Kp}");
        assert!((hf - H).abs() < 1e-5 * H.abs().max(1.0), "{hf} {H}");
        // sheet_rankine: the image sum, the direct term left out for the own patch
        let (Ha, Ka) = sheet_rankine(f64::INFINITY, false, &x, &nx, &y, &ny, false, false);
        let (Hs, Ks) = sheet_rankine(f64::INFINITY, false, &x, &nx, &y, &ny, true, false);
        assert!((Ha - Hs - H).abs() <= 1e-12 * H.abs());
        assert!((Ka - Ks - Kp).abs() <= 1e-12 * Kp.abs());
        // a plate at omega = inf: x and y in z = 0, normals -z; the odd free-surface image equals the
        // direct term (H doubled, Kp 0), which no_fs leaves out with it
        let (xp, yq, nz) = (v3(0.3, 0.1, 0.0), v3(-0.2, 0.4, 0.0), v3(0.0, 0.0, -1.0));
        let (Hd, Kd) = rankine_pair(&xp, &nz, &yq, &nz);
        let (Hi, Ki) = sheet_rankine(f64::INFINITY, true, &xp, &nz, &yq, &nz, false, false);
        assert!((Hi - 2.0 * Hd).abs() <= 1e-12 * Hd.abs() && Kd == 0.0 && Ki == 0.0);
        assert_eq!(sheet_rankine(f64::INFINITY, true, &xp, &nz, &yq, &nz, true, true), (0.0, 0.0));
        let (Hf, _) = sheet_rankine(4.0, true, &xp, &nz, &yq, &nz, false, false);
        let (Hg, _) = sheet_rankine(4.0, true, &xp, &nz, &yq, &nz, true, true);
        assert!((Hf - Hg - 2.0 * Hd).abs() <= 1e-12 * Hd.abs());
    }
}
