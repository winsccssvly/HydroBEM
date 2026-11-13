//! The source formulation of the low-order method (with walls, interior surfaces and lids,
//! pressure surfaces and in a channel): source strengths of
//! the solved potentials, the fluid velocity at the field points and on the body, the body
//! pressure and velocity at the output points and the mean drift by pressure integration
//! (theory §4.1, §11.2).

use num_complex::{Complex, Complex32, Complex64};
use rayon::prelude::*;

use crate::constants::PI;
use crate::control_drift::ControlSurfaceDrift;
use crate::error::Result;
use crate::geometry::{Vec3, length_tol, norm, v3};
use crate::green_fin::{FinPoint, fin_point};
use crate::lo_static::source_row;
use crate::low_order::{
    FreqCtx, Geometry, Output, Plan, Profile, Real, SourceInput, Static, SurfaceKind, apply_kernel, channel_grad,
    channel_influence, incident, pair_kernel,
};
use crate::parallel::{Block, in_batches, par_rows, write_blocks};
use crate::plate_geometry::{PlateIndex, Tie};
use crate::polygon::P2;
use crate::pressure_drift::{ModeShapes, PressureDrift};
use crate::problem::{Body, Problem};
use crate::progress::Work;
use crate::quadratic_qtf::{CsRecorder, QtfRecorder};
use crate::tanks::add_tank_drift;

const CZ: Complex64 = Complex64::new(0.0, 0.0);

/// `Complex32(z)` of a Complex64: each part rounded to float.
#[inline]
pub(crate) fn to_cf(z: Complex64) -> Complex32 {
    Complex32::new(z.re as f32, z.im as f32)
}

// A panel edge of the body waterline: both ends in z = 0, its midpoint and length.
struct Wl {
    j: usize,
    x: Vec3,
    dl: f64,
}

/// The potential at a waterline point x from the potential phi and velocity v at the centroid c of
/// its panel, the first-order Taylor expansion phi + (x − c)·v (theory §11.4). The centroid of the
/// waterline panel lies half its height below z = 0, where phi differs from its waterline value by
/// about K z_c (the free-surface condition phi_z = K phi): a difference of first order in the panel
/// height, which on a mesh graded towards the waterline is the largest term of the pressure drift's
/// error; the expansion leaves an error of second order in the height. With the diagnostic hook
/// HB_WATERLINE_CENTROID (`env_hooks`) the centroid value phi itself, unexpanded.
pub(crate) fn at_waterline(phi: Complex64, v: &[Complex64; 3], x: &Vec3, c: &Vec3) -> Complex64 {
    if crate::env_hooks::waterline_centroid() {
        return phi;
    }
    phi + (x[0] - c[0]) * v[0] + (x[1] - c[1]) * v[1] + (x[2] - c[2]) * v[2]
}

// The waterline edges of the wetted panels, of the stored panels and their reflections (images
// false: no wall images) or of every panel (true).
fn waterline_scan(g: &Geometry, images: bool) -> Vec<Wl> {
    let mut wl = Vec::new();
    for j in 0..g.Nf as usize {
        if g.kind[j] != SurfaceKind::Wet || (!images && g.mirror[j] != 0) {
            continue;
        }
        // the vertices as given: the projection of a warped panel moves them off z = 0; a
        // panel lying in z = 0 has no waterline
        let v = &g.raw[j];
        let tol = length_tol(g.pg[j].radius);
        if v.iter().all(|q| q[2].abs() <= tol) {
            continue;
        }
        for e in 0..4 {
            let a = v[e];
            let c = v[(e + 1) % 4];
            if a[2].abs() <= tol && c[2].abs() <= tol && norm(c - a) > tol {
                wl.push(Wl { j, x: 0.5 * (a + c), dl: norm(c - a) });
            }
        }
    }
    wl
}

// The body waterline without the wall images.
fn waterline_edges(g: &Geometry) -> Vec<Wl> {
    waterline_scan(g, false)
}

/// For every waterline element the free-surface plate panel over the water beside it (the plate's
/// edge on the element's wall: the free surface replaced by the plate there), -1 under a free
/// surface: the nearest plate panel (not a wall image) to the point a twentieth of the element's
/// length from its midpoint into the water, within a fortieth of that length or a tenth of the
/// panel's longest side (the chords of two meshes of one curve). The route admits these drifts only
/// when every plate edge lies on a wall or on another plate element (plate_rules::free_plate_edge,
/// the same tenth).
fn waterline_plates(g: &Geometry, wl: &[Wl]) -> Vec<i32> {
    let plates: Vec<(usize, Vec<P2>, f64)> = (0..g.Nf as usize)
        .filter(|&j| g.kind[j] == SurfaceKind::Plate && g.mirror[j] == 0)
        .map(|j| {
            let l: Vec<P2> = g.raw[j].iter().map(|v| [v[0], v[1]]).collect();
            let side = (0..l.len()).map(|k| polygon_side(&l, k)).fold(0.0, f64::max);
            (j, l, side)
        })
        .collect();
    let plates = PlateIndex::new(plates);
    wl.iter()
        .map(|e| {
            let n = &g.pg[e.j].n;
            plates.beside([e.x[0], e.x[1]], [n[0], n[1]], e.dl, Tie::First).map_or(-1, |i| plates.elements[i].0 as i32)
        })
        .collect()
}

// the length of side k of a closed loop
fn polygon_side(l: &[P2], k: usize) -> f64 {
    let (a, b) = (l[k], l[(k + 1) % l.len()]);
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// For every interior free-surface panel the wetted panel of its body with a waterline edge
/// nearest to it in the plane (-1 for the other panels): the source formulation prescribes on the
/// interior free surface the vertical velocity V = K phi of the exterior flow at the waterline, so
/// that it is continuous there; a jump makes the source strength logarithmically singular at the
/// waterline and the tangential velocity of the body panels next to it unbounded (Lee, Newman &
/// Zhu 1996, §3 and eq. 38).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation over several arrays
fn interior_waterline_panels(g: &Geometry) -> Vec<i32> {
    let Nf = g.Nf as usize;
    // the wetted panels with an edge in z = 0 (every reflection and wall image), and the
    // midpoints of those edges
    let wl = waterline_scan(g, true);
    let mut near = vec![-1i32; Nf];
    for j in 0..Nf {
        if g.kind[j] != SurfaceKind::Interior {
            continue;
        }
        let c = g.pg[j].c;
        let mut best = f64::INFINITY;
        for &Wl { j: jw, x, .. } in &wl {
            if g.body[jw] != g.body[j] {
                continue;
            }
            let d = (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2);
            if d < best {
                best = d;
                near[j] = jw as i32;
            }
        }
    }
    near
}

// Frame to body axes: frame -> global (g.yaw, g.px, g.py) -> body (yaw, position).
struct BodyAxes<'a> {
    g: &'a Geometry,
    b: &'a Body,
    cf0: f64,
    sf0: f64,
    cb: f64,
    sb: f64,
}

impl<'a> BodyAxes<'a> {
    fn new(g: &'a Geometry, b: &'a Body) -> Self {
        BodyAxes {
            g,
            b,
            cf0: (g.yaw * PI / 180.0).cos(),
            sf0: (g.yaw * PI / 180.0).sin(),
            cb: (b.yaw * PI / 180.0).cos(),
            sb: (b.yaw * PI / 180.0).sin(),
        }
    }
    fn vec(&self, v: &Vec3) -> Vec3 {
        let gx = self.cf0 * v[0] - self.sf0 * v[1];
        let gy = self.sf0 * v[0] + self.cf0 * v[1];
        v3(self.cb * gx + self.sb * gy, -self.sb * gx + self.cb * gy, v[2])
    }
    fn vecc(&self, v: &[Complex64; 3], o: &mut [Complex64; 3]) {
        let gx = self.cf0 * v[0] - self.sf0 * v[1];
        let gy = self.sf0 * v[0] + self.cf0 * v[1];
        o[0] = self.cb * gx + self.sb * gy;
        o[1] = -self.sb * gx + self.cb * gy;
        o[2] = v[2];
    }
    fn pt(&self, x: &Vec3) -> Vec3 {
        let gx = self.g.px + self.cf0 * x[0] - self.sf0 * x[1] - self.b.position[0];
        let gy = self.g.py + self.sf0 * x[0] + self.cf0 * x[1] - self.b.position[1];
        v3(self.cb * gx + self.sb * gy, -self.sb * gx + self.cb * gy, x[2] - self.b.position[2])
    }
}

/// The nine values of a body at offset o of an output array.
fn nine(v: &mut [f64], o: usize) -> &mut [f64; 9] {
    (&mut v[o..o + 9]).try_into().unwrap()
}

// The source-formulation outputs of one ordinary frequency. The steps of source_outputs (below)
// are its member functions, in their order; they share the source strengths of the solved
// potentials and the velocities at the panels.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct Sources<'a, R: Real> {
    m: &'a Problem,
    g: &'a Geometry,
    pl: &'a Plan,
    sf: &'a Static<R>,
    fc: &'a FreqCtx,
    fi: i32,
    work: Work<'a>,
    in_: &'a SourceInput<'a, R>,
    prof: &'a mut Profile,
    out: &'a mut Output,
    phi: &'a [Complex64],
    u: &'a [Complex64],
    phiI: &'a [Complex64],
    dphiI: &'a [Complex64],
    LphiI: &'a [Complex64],
    Smat: &'a [Complex<R>],
    Vst: &'a [[Complex<R>; 3]],
    fpt: &'a [FinPoint],
    w: f64,
    Ns: i32,
    Nf: i32,
    G: i32,
    NC: i32,
    nrad: i32,
    nh: i32,
    N2: usize,
    D: i32,
    rho: f64,
    nsr: i32,           // sources of the radiation columns and the scattered potentials
    us: Vec<Complex64>, // [nsr][Nf] the boundary values and the source strengths
    sig: Vec<Complex64>,
    gj: Vec<i32>,             // the panel g j of every panel j (Geometry::panel_images)
    vel: Vec<[Complex64; 3]>, // [nsr][Nf] the velocity at every panel
}

impl<'a, R: Real> Sources<'a, R> {
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    fn new(
        m: &'a Problem,
        g: &'a Geometry,
        pl: &'a Plan,
        sf: &'a Static<R>,
        fc: &'a FreqCtx,
        fi: i32,
        work: Work<'a>,
        in_: &'a SourceInput<'a, R>,
        prof: &'a mut Profile,
        out: &'a mut Output,
    ) -> Self {
        let nrad = pl.rad.len() as i32;
        let nh = if pl.diffraction { pl.heads.len() as i32 } else { 0 };
        Sources {
            m,
            g,
            pl,
            sf,
            fc,
            fi,
            work,
            in_,
            prof,
            out,
            phi: in_.phi,
            u: in_.u,
            phiI: in_.phiI,
            dphiI: in_.dphiI,
            LphiI: in_.LphiI,
            Smat: in_.Smat,
            Vst: in_.Vst,
            fpt: in_.fpt,
            w: fc.w,
            Ns: g.Ns,
            Nf: g.Nf,
            G: g.G,
            NC: g.chi.len() as i32,
            nrad,
            nh,
            N2: g.Ns as usize * g.Ns as usize,
            D: pl.D,
            rho: pl.rho,
            nsr: nrad + nh,
            us: Vec::new(),
            sig: Vec::new(),
            gj: Vec::new(),
            vel: Vec::new(),
        }
    }

    // grad_x of the source integral of panel j at the centroid of sector panel k (Vst)
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn V(&self, k: usize, j: usize) -> [Complex64; 3] {
        let e = &self.Vst[k * self.Nf as usize + j];
        [R::c_as_cd(e[0]), R::c_as_cd(e[1]), R::c_as_cd(e[2])]
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn source_strengths(&mut self) -> Result<()> {
        let (g, fc) = (self.g, self.fc);
        let (Ns, Nf, G, NC, N2, nrad, nh, nsr) = (
            self.Ns as usize,
            self.Nf as usize,
            self.G as usize,
            self.NC as usize,
            self.N2,
            self.nrad as usize,
            self.nh as usize,
            self.nsr as usize,
        );
        self.us = vec![CZ; nsr * Nf];
        for c in 0..nrad {
            for j in 0..Nf {
                self.us[c * Nf + j] = self.u[c * Nf + j];
            }
        }
        // the interior free surface: V = K phi of the exterior flow at the waterline (the potential
        // formulation's solution at the nearest wetted waterline panel); the scattered potential
        // for the diffraction columns
        let near = interior_waterline_panels(g);
        for j in 0..Nf {
            if near[j] < 0 {
                continue;
            }
            let jw = near[j] as usize;
            for c in 0..nrad {
                self.us[c * Nf + j] = fc.K * self.phi[c * Nf + jw];
            }
        }
        for h in 0..nh {
            for j in 0..Nf {
                // the sources carry the scattered potential: -dphi_I/dn on the body; on a lid
                // the condition of the total potential, phi_Sz - kappa phi_S = (kappa - K) phi_I
                let mut v = CZ;
                if g.body_surface(j as i32) {
                    v = -self.dphiI[h * Nf + j];
                } else if g.kind[j] == SurfaceKind::Lid {
                    v = (fc.kappa[g.lid[j] as usize] - fc.K) * self.phiI[h * Nf + j];
                } else if g.kind[j] == SurfaceKind::Press {
                    v = self.LphiI[h * Nf + j]; // L phi_S = -L phi_I
                } else if near[j] >= 0 {
                    let jw = near[j] as usize;
                    let mut ps = self.phi[(nrad + h) * Nf + jw];
                    if self.pl.total {
                        ps -= self.phiI[h * Nf + jw];
                    }
                    v = fc.K * ps;
                }
                self.us[(nrad + h) * Nf + j] = v;
            }
        }
        self.gj = g.panel_images(); // the panel g j of every panel j
        // the source strengths per character: -2 pi sigma_k + sum K'(k, n) sigma_n = u_k
        self.sig = vec![CZ; nsr * Nf];
        {
            // rows in z = 0, the limit from below with the upward normal: G
            // satisfies dG/dz = K G there for every source panel but the own one, whose
            // jump is 4 pi sigma, so d phi/dz = 4 pi sigma + K phi. Interior surfaces:
            // d phi/dz = V (the extended equation of theory §5.1 with a Neumann condition inside
            // the body; V = K phi at the waterline, see interior_waterline_panels), lids:
            // d phi/dz = kappa phi, i.e. 4 pi sigma + (K - kappa) phi = 0.
            // Pressure surfaces, L phi = phi_z - K phi = -u: in z = 0 by the same
            // identity -4 pi sigma_k = u_k (no coupling); submerged (normal down, into the
            // fluid) n . grad phi + K phi = u, the body row plus K S. Free-surface plates, normal
            // down: dphi/dn = -(4 pi sigma + K phi) = u, the row -4 pi sigma_k - K sum S sigma = u_k
            let mut Km = vec![Complex::<R>::default(); NC * N2];
            {
                let this = &*self;
                let work = self.work;
                // the rows on the inner threads, each row k of every character a block of its own
                let rows: Vec<usize> = (0..Ns).collect();
                let row = |&k: &usize| -> Result<Block<Complex<R>>> {
                    work.check_cancel()?;
                    let mut b = Block::new(k, 1, 0, Ns, NC, Complex::<R>::default());
                    let jk = g.sec[k];
                    let nk = &g.pg[jk as usize].n;
                    let kk = g.kind[jk as usize];
                    let pz0 = g.press_z0(jk);
                    let psub = kk == SurfaceKind::Press && !pz0;
                    let free_k = if kk == SurfaceKind::Interior || kk == SurfaceKind::Lid {
                        4.0 * PI
                    } else if pz0 || kk == SurfaceKind::Plate {
                        -4.0 * PI
                    } else {
                        -2.0 * PI
                    };
                    for n in 0..Ns {
                        let mut kc = [CZ; 4];
                        if pz0 {
                            // no coupling
                        } else if kk == SurfaceKind::Lid || kk == SurfaceKind::Interior || kk == SurfaceKind::Plate {
                            let c = if kk == SurfaceKind::Lid {
                                fc.K - fc.kappa[g.lid[jk as usize] as usize]
                            } else if kk == SurfaceKind::Plate {
                                Complex64::new(-fc.K, 0.0)
                            } else {
                                Complex64::new(fc.K, 0.0)
                            };
                            for ch in 0..NC {
                                kc[ch] = c * R::c_as_cd(this.Smat[ch * N2 + k + n * Ns]);
                            }
                        } else {
                            for gi in 0..G {
                                let v = this.V(k, g.img[gi * Ns + n] as usize);
                                let kn = v[0] * nk[0] + v[1] * nk[1] + v[2] * nk[2];
                                for ch in 0..NC {
                                    kc[ch] += g.chi[ch][gi] * kn;
                                }
                            }
                            if psub {
                                for ch in 0..NC {
                                    kc[ch] += fc.K * R::c_as_cd(this.Smat[ch * N2 + k + n * Ns]);
                                }
                            }
                        }
                        for ch in 0..NC {
                            b.set(
                                ch,
                                k,
                                n,
                                R::c_from(kc[ch] + (if k == n { Complex64::new(free_k, 0.0) } else { CZ })),
                            );
                        }
                    }
                    Ok(b)
                };
                let mut mats: Vec<&mut [Complex<R>]> = Km.chunks_mut(N2.max(1)).collect();
                in_batches(&rows, 16 * rayon::current_num_threads(), row, |blocks| {
                    write_blocks(&mut mats, Ns, &blocks.iter().collect::<Vec<_>>());
                })?;
            }
            let mut A2 = vec![Complex::<R>::default(); N2];
            let mut rhs2 = vec![Complex::<R>::default(); Ns * nsr];
            for ch in 0..NC {
                let chi = &g.chi[ch];
                for c in 0..nsr {
                    for n in 0..Ns {
                        let mut v = CZ;
                        for gi in 0..G {
                            v += chi[gi] * self.us[c * Nf + g.img[gi * Ns + n] as usize];
                        }
                        rhs2[n + c * Ns] = R::c_from(v / G as f64);
                    }
                }
                A2.copy_from_slice(&Km[ch * N2..(ch + 1) * N2]);
                R::lu_solve(Ns as i32, &mut A2, nsr as i32, &mut rhs2, self.work)?;
                for c in 0..nsr {
                    for gi in 0..G {
                        for n in 0..Ns {
                            self.sig[c * Nf + g.img[gi * Ns + n] as usize] += chi[gi] * R::c_as_cd(rhs2[n + c * Ns]);
                        }
                    }
                }
            }
        }
        self.prof.mark("  source strengths");
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn field_velocities(&mut self) -> Result<()> {
        let (g, pl, fc, sf) = (self.g, self.pl, self.fc, self.sf);
        let (Nf, nrad, nh, nsr, w) =
            (self.Nf as usize, self.nrad as usize, self.nh as usize, self.nsr as usize, self.w);
        let (sig, fpt) = (&self.sig, self.fpt);
        // velocities at the field points: sum_j sigma_j grad_x int_j G, in global axes
        let Pf = g.fp.len();
        let cy = (g.yaw * PI / 180.0).cos();
        let sy = (g.yaw * PI / 180.0).sin();
        let global = |v: &[Complex64; 3], o: &mut [Complex32]| {
            o[0] = to_cf(cy * v[0] - sy * v[1]);
            o[1] = to_cf(sy * v[0] + cy * v[1]);
            o[2] = to_cf(v[2]);
        };
        let c2f = 2.0 * fc.K * fc.logc;
        let work = self.work;
        let out = &mut *self.out;
        // the points on the inner threads, each its own outputs
        let rows = par_rows(&mut out.fvr, 3 * nrad, Pf)
            .zip(par_rows(&mut out.fvi, 3 * nh, Pf))
            .zip(par_rows(&mut out.fvs, 3 * nh, Pf).zip(par_rows(&mut out.fvt, 3 * nh, Pf)));
        rows.enumerate().try_for_each(|(q, ((fvr, fvi), (fvs, fvt)))| -> Result<()> {
            work.check_cancel()?;
            if q < sf.vfrom {
                return Ok(()); // no velocity asked for (left zero)
            }
            let x = &g.fp[q];
            let fx = if fc.finite() { fin_point(&fc.fd, x[2]) } else { FinPoint::default() };
            let mut vr = vec![CZ; nsr * 3];
            // the velocity rows of this point made now (rows too large to store) or the stored ones
            let (own_g, own_l) = match &sf.lazy {
                Some(lz) => {
                    let mut a = vec![[R::default(); 3]; Nf];
                    let mut b = vec![[R::default(); 3]; Nf];
                    source_row(g, x, lz.logint, lz.tol, lz.depth, &mut a, &mut b);
                    (a, b)
                }
                None => (Vec::new(), Vec::new()),
            };
            let (rg, rl, base): (&[[R; 3]], &[[R; 3]], usize) = match &sf.lazy {
                Some(_) => (&own_g, &own_l, 0),
                None => (&sf.vg, &sf.vl, (q - sf.vfrom) * Nf),
            };
            for j in 0..Nf {
                let kj = g.kind[j];
                if kj != SurfaceKind::Wet
                    && kj != SurfaceKind::Interior
                    && kj != SurfaceKind::Lid
                    && kj != SurfaceKind::Press
                    && kj != SurfaceKind::Plate
                {
                    continue;
                }
                let o = (q - sf.vfrom) * Nf + j; // the velocity rows of point q
                if g.W > 0.0 {
                    // channel
                    let vg: [&[[R; 3]]; 3] = [&sf.vg, &sf.vg1, &sf.vg2];
                    let vl: [&[[R; 3]]; 3] = [&sf.vl, &sf.vl1, &sf.vl2];
                    let v = channel_grad(g, vg, vl, o, j as i32, x, &fx, &fpt[j], fc);
                    for c in 0..nsr {
                        let s = sig[c * Nf + j];
                        for d in 0..3 {
                            vr[3 * c + d] += s * v[d];
                        }
                    }
                    continue;
                }
                let (wk, Rr) = if fc.finite() {
                    pair_kernel(fc, x, &g.pg[j].c, Some(&fx), Some(&fpt[j]))
                } else {
                    pair_kernel(fc, x, &g.pg[j].c, None, None)
                };
                let Aj = g.pg[j].area;
                let ax = if Rr > 0.0 { (x[0] - g.pg[j].c[0]) / Rr } else { 0.0 };
                let ay = if Rr > 0.0 { (x[1] - g.pg[j].c[1]) / Rr } else { 0.0 };
                let wr = [wk.WrR * ax, wk.WrR * ay, wk.Wrz];
                let wi = [wk.WiR * ax, wk.WiR * ay, wk.Wiz];
                let mut v = [CZ; 3];
                for d in 0..3 {
                    let ov = base + j;
                    v[d] = Complex64::new(rg[ov][d].as_f64() + Aj * wr[d] - c2f * rl[ov][d].as_f64(), Aj * wi[d]);
                }
                for c in 0..nsr {
                    let s = sig[c * Nf + j];
                    for d in 0..3 {
                        vr[3 * c + d] += s * v[d];
                    }
                }
            }
            let iw = Complex64::new(0.0, w);
            for c in 0..nrad {
                let v = [iw * vr[3 * c], iw * vr[3 * c + 1], iw * vr[3 * c + 2]]; // per unit displacement
                global(&v, &mut fvr[c * 3..c * 3 + 3]);
            }
            for h in 0..nh {
                let (_, gI) = incident(g, pl, fc, pl.heads[h], x);
                let vs = [vr[3 * (nrad + h)], vr[3 * (nrad + h) + 1], vr[3 * (nrad + h) + 2]];
                let vt = [gI[0] + vs[0], gI[1] + vs[1], gI[2] + vs[2]];
                let o = h * 3;
                global(&gI, &mut fvi[o..o + 3]);
                global(&vs, &mut fvs[o..o + 3]);
                global(&vt, &mut fvt[o..o + 3]);
            }
            Ok(())
        })?;
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn panel_velocities(&mut self) {
        let g = self.g;
        let (Ns, Nf, G, nsr) = (self.Ns as usize, self.Nf as usize, self.G as usize, self.nsr as usize);
        let nrad = self.nrad as usize;
        // velocities at every panel: at img(g, k), R_g [-2 pi sigma n_k + sum_j V(k, j) sigma(g j)]
        // (g j: the panel img(g g', n) of j = img(g', n)); blocks of sector rows by matrix products
        self.vel = vec![[CZ; 3]; nsr * Nf];
        {
            let ncol = G * nsr;
            let mut Sg = vec![CZ; Nf * ncol]; // [j][gi * nsr + c] = sigma_c(g j), column-major Nf x ncol
            for gi in 0..G {
                for c in 0..nsr {
                    for jj in 0..Nf {
                        let t = self.gj[gi * Nf + jj];
                        if t >= 0 {
                            Sg[jj + (gi * nsr + c) * Nf] = self.sig[c * Nf + t as usize];
                        }
                    }
                }
            }
            const BLOCK: usize = 32;
            let mut Vb = vec![CZ; 3 * BLOCK * Nf];
            let mut Pb = vec![CZ; 3 * BLOCK * ncol];
            let mut k0 = 0;
            while k0 < Ns {
                let nb = BLOCK.min(Ns - k0);
                let M = 3 * nb;
                // V rows (k, d) of the block, row-major in the column-major sense: Vb[(3 kk + d) + j M]
                for kk in 0..nb {
                    for jj in 0..Nf {
                        let v = self.V(k0 + kk, jj);
                        for d in 0..3 {
                            Vb[3 * kk + d + jj * M] = v[d];
                        }
                    }
                }
                // Vb is M x Nf (ld M), Sg Nf x ncol (ld Nf), Pb M x ncol (ld M)
                crate::linalg::gemm(M, ncol, Nf, &Vb, M, &Sg, Nf, &mut Pb, M);
                for kk in 0..nb {
                    let k = k0 + kk;
                    let nk = &g.pg[g.sec[k] as usize].n;
                    for gi in 0..G {
                        let t = g.img[gi * Ns + k] as usize;
                        for c in 0..nsr {
                            let pr = &Pb[3 * kk + (gi * nsr + c) * M..];
                            let sk = -2.0 * PI * self.sig[c * Nf + t];
                            let v = [pr[0] + sk * nk[0], pr[1] + sk * nk[1], pr[2] + sk * nk[2]];
                            self.vel[c * Nf + t] = [g.refl[gi].sx * v[0], g.refl[gi].sy * v[1], v[2]];
                            if g.plate(t as i32) {
                                // a free-surface plate coincides with its free-surface image as a
                                // pressure surface in z = 0 does: its vertical velocity is that of its
                                // boundary condition, -dphi/dn = -u (the scattered part for diffraction)
                                self.vel[c * Nf + t][2] = -self.us[c * Nf + t];
                            }
                            if g.press_z0(t as i32) {
                                // a pressure panel in z = 0 coincides with its free-surface image:
                                // the vertical derivative of the regular part of the own kernel
                                // keeps a log singularity, so the vertical velocity comes from the
                                // identity of the source rows, phi_z = K phi + 4 pi sigma (phi the
                                // potential of the sources: the scattered one for diffraction); the
                                // horizontal components are the integrals above (the own panel
                                // contributes its in-plane principal values)
                                let mut ps = self.phi[c * Nf + t];
                                if c >= nrad && self.pl.total {
                                    ps -= self.phiI[(c - nrad) * Nf + t];
                                }
                                self.vel[c * Nf + t][2] = self.fc.K * ps + 4.0 * PI * self.sig[c * Nf + t];
                            }
                        }
                    }
                }
                k0 += BLOCK;
            }
        }
        self.prof.mark("  body velocities");
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn body_velocities(&mut self) {
        let (g, pl, fc) = (self.g, self.pl, self.fc);
        let (Nf, nrad, nh, w) = (self.Nf as usize, self.nrad as usize, self.nh as usize, self.w);
        // the body velocities in global axes: radiation per unit displacement, total diffraction
        let cy = (g.yaw * PI / 180.0).cos();
        let sy = (g.yaw * PI / 180.0).sin();
        let Nb = pl.bp.len();
        let put3 = |v: &[Complex64; 3], o: &mut [Complex32]| {
            o[0] = to_cf(cy * v[0] - sy * v[1]);
            o[1] = to_cf(sy * v[0] + cy * v[1]);
            o[2] = to_cf(v[2]);
        };
        for q in 0..Nb {
            let j = pl.bp[q] as usize;
            for c in 0..nrad {
                let vr = &self.vel[c * Nf + j];
                let v =
                    [Complex64::new(0.0, w) * vr[0], Complex64::new(0.0, w) * vr[1], Complex64::new(0.0, w) * vr[2]];
                let o = (q * nrad + c) * 3;
                put3(&v, &mut self.out.bvr[o..o + 3]);
            }
            for h in 0..nh {
                let (_, gI) = incident(g, pl, fc, pl.heads[h], &g.pg[j].c);
                let vs = &self.vel[(nrad + h) * Nf + j];
                let v = [gI[0] + vs[0], gI[1] + vs[1], gI[2] + vs[2]];
                let o = (q * nh + h) * 3;
                put3(&v, &mut self.out.bvd[o..o + 3]);
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn pressure_drift(&mut self) {
        let (m, g, pl, fc) = (self.m, self.g, self.pl, self.fc);
        let (Nf, nrad, nh, D, rho, w, fi) = (
            self.Nf as usize,
            self.nrad as usize,
            self.nh as usize,
            self.D as usize,
            self.rho,
            self.w,
            self.fi as usize,
        );
        let (phi, vel) = (self.phi, &self.vel);
        let gr = pl.grav;
        let iw = Complex64::new(0.0, w);
        let wl = waterline_edges(g);
        let under = waterline_plates(g, &wl);
        let plated = under.iter().any(|&p| p >= 0);
        let NB = m.bodies.len();
        // the generalized modes: the first of each body among all, their shapes at the panels'
        // body-local centroids and at the waterline elements (theory §10.1)
        let gfirst: Vec<usize> =
            m.bodies.iter().scan(0, |s, b| Some(std::mem::replace(s, *s + b.generalized.len()))).collect();
        let GT: usize = m.bodies.iter().map(|b| b.generalized.len()).sum();
        let shapes = |bi: usize, x: &Vec3, rec: i32| -> (Vec<Vec3>, Vec<[[f64; 3]; 3]>) {
            let gm = &m.bodies[bi].generalized;
            (gm.iter().map(|q| q.displacement(x, rec)).collect(), gm.iter().map(|q| q.gradient(x, rec)).collect())
        };
        // the first-order values of the quadratic QTFs, in the order of the drift
        let mut rec = if pl.pqtf.any() { Some(QtfRecorder::new(w, nh)) } else { None };
        for h in 0..nh {
            if let Some(r) = rec.as_mut() {
                r.heading(h);
            }
            let mut Xr = vec![CZ; nrad]; // the motions of every radiation column
            if pl.rao {
                for r in 0..nrad {
                    let v = self.out.M[h * D + pl.rad[r].coord as usize];
                    Xr[r] = Complex64::new(v.re as f64, v.im as f64);
                }
            }
            // the total potential and velocity at panel j (frame), with the motions of all bodies
            let field = |j: usize, ph: &mut Complex64, gv: &mut [Complex64; 3]| {
                let (pI, gI) = incident(g, pl, fc, pl.heads[h], &g.pg[j].c);
                *ph = if pl.total { phi[(nrad + h) * Nf + j] } else { phi[(nrad + h) * Nf + j] + pI };
                let vs = &vel[(nrad + h) * Nf + j];
                for d in 0..3 {
                    gv[d] = gI[d] + vs[d];
                }
                for r in 0..nrad {
                    let x = iw * Xr[r];
                    if x == CZ {
                        continue;
                    }
                    *ph += x * phi[r * Nf + j];
                    let vr = &vel[r * Nf + j];
                    for d in 0..3 {
                        gv[d] += x * vr[d];
                    }
                }
            };
            // the six rigid motions of a body and those of its generalized modes (always free)
            let rigid = |bi: usize| {
                let mut X = [CZ; 6];
                for r in 0..nrad {
                    if pl.rad[r].body == bi as i32 && pl.rad[r].mode > 0 {
                        X[(pl.rad[r].mode - 1) as usize] = Xr[r];
                    }
                }
                X
            };
            let modal = |bi: usize| {
                let mut xg = vec![CZ; m.bodies[bi].generalized.len()];
                for r in 0..nrad {
                    if pl.rad[r].body == bi as i32 && pl.rad[r].mode == -1 {
                        xg[pl.rad[r].gm as usize - gfirst[bi]] = Xr[r];
                    }
                }
                xg
            };
            // the motions of every body for the plate edges over the waterline (the band under a
            // plate rises with the plate's edge, of the wall's body or another)
            let motions: Vec<PressureDrift> = if plated {
                (0..NB)
                    .map(|bi| {
                        let b = &m.bodies[bi];
                        let mut q = PressureDrift::new(&rigid(bi), &b.rotation_center, rho, gr, w, false);
                        if !b.generalized.is_empty() {
                            q.generalized(&modal(bi));
                        }
                        q
                    })
                    .collect()
            } else {
                Vec::new()
            };
            for bi in 0..NB {
                let b = &m.bodies[bi];
                let X = rigid(bi);
                let ax = BodyAxes::new(g, b);
                // with pressure surfaces the hydrostatic part by the closed forms
                let closed = !b.pressure.is_empty();
                let mut pd = PressureDrift::new(&X, &b.rotation_center, rho, gr, w, closed);
                if let Some(r) = rec.as_mut() {
                    r.group(bi, false, rho, closed, &b.rotation_center, b.position[2], &X);
                }
                let ng = b.generalized.len();
                let gmodes = ng > 0;
                if gmodes {
                    let xg = modal(bi);
                    pd.generalized(&xg);
                    if let Some(r) = rec.as_mut() {
                        r.generalized(&xg);
                    }
                }
                // the mean wetted surface with the free-surface plates (P0 = 0 on them)
                for j in 0..Nf {
                    if (g.kind[j] != SurfaceKind::Wet && g.kind[j] != SurfaceKind::Plate)
                        || g.mirror[j] != 0
                        || g.body[j] != bi as i32
                    {
                        continue;
                    }
                    let mut ph = CZ;
                    let mut gvf = [CZ; 3];
                    let mut gv = [CZ; 3];
                    field(j, &mut ph, &mut gvf);
                    ax.vecc(&gvf, &mut gv);
                    let (xb, n) = (ax.pt(&g.pg[j].c), ax.vec(&g.pg[j].n));
                    if gmodes {
                        let (s, bg) = shapes(bi, &g.lc[j], g.rec[j] + 1);
                        let sh = ModeShapes { s: &s, b: &bg };
                        pd.surface_with(ph, &gv, &xb, &n, g.pg[j].c[2], g.pg[j].area, Some(&sh));
                        if let Some(r) = rec.as_mut() {
                            r.surface_with(ph, &gv, &xb, &n, g.pg[j].c[2], g.pg[j].area, Some(&sh));
                        }
                        continue;
                    }
                    pd.surface(ph, &gv, &xb, &n, g.pg[j].c[2], g.pg[j].area);
                    if let Some(r) = rec.as_mut() {
                        r.surface(ph, &gv, &xb, &n, g.pg[j].c[2], g.pg[j].area);
                    }
                }
                for (e, &jp) in wl.iter().zip(&under) {
                    if g.body[e.j] != bi as i32 {
                        continue;
                    }
                    let mut ph = CZ;
                    let mut gv = [CZ; 3];
                    field(e.j, &mut ph, &mut gv);
                    ph = at_waterline(ph, &gv, &e.x, &g.pg[e.j].c);
                    let (xb, n) = (ax.pt(&e.x), ax.vec(&g.pg[e.j].n));
                    if jp >= 0 {
                        // under a plate: the vertical displacement of the plate's edge there, of
                        // its body (this one or another) and of its generalized modes
                        let jp = jp as usize;
                        let bp = g.body[jp] as usize;
                        let xp = BodyAxes::new(g, &m.bodies[bp]).pt(&e.x);
                        let sp = (!m.bodies[bp].generalized.is_empty()).then(|| shapes(bp, &xp, g.rec[jp] + 1).0);
                        let top = motions[bp].vertical(&xp, sp.as_deref());
                        let s = gmodes.then(|| shapes(bi, &xb, g.rec[e.j] + 1).0);
                        pd.waterline_under_plate(ph, top, &xb, &n, e.dl, s.as_deref());
                        if let Some(r) = rec.as_mut() {
                            r.waterline_under_plate(ph, top, &xb, &n, e.dl, s.as_deref());
                        }
                        continue;
                    }
                    if gmodes {
                        let (s, _) = shapes(bi, &xb, g.rec[e.j] + 1);
                        pd.waterline_with(ph, &xb, &n, e.dl, Some(&s));
                        if let Some(r) = rec.as_mut() {
                            r.waterline_with(ph, &xb, &n, e.dl, Some(&s));
                        }
                        continue;
                    }
                    pd.waterline(ph, &xb, &n, e.dl);
                    if let Some(r) = rec.as_mut() {
                        r.waterline(ph, &xb, &n, e.dl);
                    }
                }
                let mut crg = [0.0f64; 36];
                if closed {
                    for j in 0..Nf {
                        if g.mirror[j] == 0
                            && g.body[j] == bi as i32
                            && (g.kind[j] == SurfaceKind::Wet || g.kind[j] == SurfaceKind::Press)
                        {
                            let (x, n, wet) = (ax.pt(&g.pg[j].c), ax.vec(&g.pg[j].n), g.kind[j] == SurfaceKind::Wet);
                            pd.hydrostatic_point(&x, &n, g.pg[j].c[2], g.pg[j].area, wet);
                            if let Some(r) = rec.as_mut() {
                                r.hydrostatic_point(&x, &n, g.pg[j].c[2], g.pg[j].area, wet);
                            }
                        }
                    }
                    let hs = m.hydrostatics(bi as i32);
                    crg = hs.c_rho_g;
                }
                let o9 = nine(&mut self.out.PD, (h * NB + bi) * 9);
                pd.finish(b.position[2], &crg, o9);
                if gmodes {
                    let o = h * GT + gfirst[bi];
                    self.out.PG[o..o + ng].copy_from_slice(pd.generalized_forces());
                }
                add_tank_drift(pl, bi as i32, fi as i32, w, &X, &b.rotation_center, &Xr, o9, rec.as_mut());
            }
        }
        if let Some(r) = rec {
            self.out.qtf[0] = r.finish();
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn control_drift(&mut self) {
        let (m, g, pl, fc) = (self.m, self.g, self.pl, self.fc);
        let (Nf, G, nrad, nh, nsr, D, rho, w) = (
            self.Nf as usize,
            self.G as usize,
            self.nrad as usize,
            self.nh as usize,
            self.nsr as usize,
            self.D as usize,
            self.rho,
            self.w,
        );
        let (phi, sig, gj, fpt) = (self.phi, &self.sig, &self.gj, self.fpt);
        // the mean drift by the control surfaces: the total potential and velocity at
        // the control-surface points from the sources, per source column
        let sc = self.in_.scs.unwrap();
        let Pc = g.cs.len();
        let Pr = g.cs_reps.len();
        let c2f = 2.0 * fc.K * fc.logc;
        // per representative r and group element gi: phi(g x_r) = sum_j sigma(g j) S(x_r, j),
        // v(g x_r) = R_g sum_j sigma(g j) grad S(x_r, j)
        let mut need = vec![false; Pr * G];
        for q in 0..Pc {
            need[g.cs_rep[q] as usize * G + g.cs_g[q] as usize] = true;
        }
        let mut pc = vec![CZ; Pr * G * nsr];
        let mut vc = vec![CZ; Pr * G * nsr * 3];
        let mut Sx = vec![CZ; Nf];
        let mut Vx = vec![CZ; Nf * 3];
        for r in 0..Pr {
            let x = &g.cs[g.cs_reps[r] as usize].x;
            let fx = if fc.finite() { fin_point(&fc.fd, x[2]) } else { FinPoint::default() };
            let row = r * Nf;
            for j in 0..Nf {
                let mut S = CZ;
                let mut v = [CZ; 3];
                let kj = g.kind[j];
                if kj == SurfaceKind::Wet
                    || kj == SurfaceKind::Interior
                    || kj == SurfaceKind::Lid
                    || kj == SurfaceKind::Press
                    || kj == SurfaceKind::Plate
                {
                    let o = row + j;
                    if g.W > 0.0 {
                        (S, _) = channel_influence(g, sc, row, j as i32, x, &fx, &fpt[j], fc);
                        let vg: [&[[R; 3]]; 3] = [&sc.vg, &sc.vg1, &sc.vg2];
                        let vl: [&[[R; 3]]; 3] = [&sc.vl, &sc.vl1, &sc.vl2];
                        v = channel_grad(g, vg, vl, o, j as i32, x, &fx, &fpt[j], fc);
                    } else {
                        let (wk, Rr) = if fc.finite() {
                            pair_kernel(fc, x, &g.pg[j].c, Some(&fx), Some(&fpt[j]))
                        } else {
                            pair_kernel(fc, x, &g.pg[j].c, None, None)
                        };
                        (S, _) = apply_kernel(g, sc, row, j as i32, x, fc, &wk, Rr, false);
                        let Aj = g.pg[j].area;
                        let ax = if Rr > 0.0 { (x[0] - g.pg[j].c[0]) / Rr } else { 0.0 };
                        let ay = if Rr > 0.0 { (x[1] - g.pg[j].c[1]) / Rr } else { 0.0 };
                        let wr = [wk.WrR * ax, wk.WrR * ay, wk.Wrz];
                        let wi = [wk.WiR * ax, wk.WiR * ay, wk.Wiz];
                        for d in 0..3 {
                            v[d] = Complex64::new(
                                sc.vg[o][d].as_f64() + Aj * wr[d] - c2f * sc.vl[o][d].as_f64(),
                                Aj * wi[d],
                            );
                        }
                    }
                }
                Vx[3 * j..3 * j + 3].copy_from_slice(&v);
                Sx[j] = S;
            }
            for gi in 0..G {
                if !need[r * G + gi] {
                    continue;
                }
                let pj = &gj[gi * Nf..(gi + 1) * Nf];
                for c in 0..nsr {
                    let mut p = CZ;
                    let mut vs = [CZ; 3];
                    for j in 0..Nf {
                        if pj[j] < 0 {
                            continue;
                        }
                        let s = sig[c * Nf + pj[j] as usize];
                        p += s * Sx[j];
                        for d in 0..3 {
                            vs[d] += s * Vx[3 * j + d];
                        }
                    }
                    let o = (r * G + gi) * nsr + c;
                    pc[o] = p;
                    vc[3 * o] = g.refl[gi].sx * vs[0];
                    vc[3 * o + 1] = g.refl[gi].sy * vs[1];
                    vc[3 * o + 2] = vs[2];
                }
            }
        }
        self.prof.mark("  control-surface points");
        let gr = pl.grav;
        let iw = Complex64::new(0.0, w);
        // the body waterline: panel edges in z = 0 bordered by free surface (under a free-surface
        // plate, whose edge lies on the wall, the free surface S_f has no boundary: no line term)
        let wl = waterline_edges(g);
        let under = waterline_plates(g, &wl);
        let NB = m.bodies.len();
        let out = &mut *self.out;
        // the control-surface values of the quadratic QTFs
        let mut rec = if pl.cqtf { Some(CsRecorder::new(nh)) } else { None };
        for h in 0..nh {
            if let Some(r) = rec.as_mut() {
                r.heading(h);
            }
            let mut Xr = vec![CZ; nrad];
            if pl.rao {
                for r in 0..nrad {
                    let v = out.M[h * D + pl.rad[r].coord as usize];
                    Xr[r] = Complex64::new(v.re as f64, v.im as f64);
                }
            }
            for bi in 0..NB {
                let b = &m.bodies[bi];
                let mut X = [CZ; 6];
                for r in 0..nrad {
                    if pl.rad[r].body == bi as i32 && pl.rad[r].mode > 0 {
                        X[(pl.rad[r].mode - 1) as usize] = Xr[r];
                    }
                }
                let ax = BodyAxes::new(g, b);
                let mut cdr =
                    ControlSurfaceDrift::new(&X, &b.rotation_center, rho, gr, w, b.control_surface_alternative);
                if let Some(r) = rec.as_mut() {
                    r.group(bi, rho, b.control_surface_alternative, &b.rotation_center, b.position[2], &X);
                }
                for q in 0..Pc {
                    let cp = &g.cs[q];
                    if cp.body != bi as i32 {
                        continue;
                    }
                    let (pI, gI) = incident(g, pl, fc, pl.heads[h], &cp.x);
                    let o0 = (g.cs_rep[q] as usize * G + g.cs_g[q] as usize) * nsr;
                    let mut ph = pI + pc[o0 + nrad + h];
                    let mut vf = [CZ; 3];
                    for d in 0..3 {
                        vf[d] = gI[d] + vc[(o0 + nrad + h) * 3 + d];
                    }
                    for r in 0..nrad {
                        let xr = iw * Xr[r];
                        if xr == CZ {
                            continue;
                        }
                        ph += xr * pc[o0 + r];
                        for d in 0..3 {
                            vf[d] += xr * vc[(o0 + r) * 3 + d];
                        }
                    }
                    let mut vb = [CZ; 3];
                    ax.vecc(&vf, &mut vb);
                    if cp.part == 0 {
                        cdr.submerged(&vb, &cp.xb, &cp.nb, cp.w);
                    } else if cp.part == 1 {
                        cdr.free_surface(ph, &vb, &cp.xb, cp.w);
                    } else {
                        cdr.line(ph, &cp.xb, &cp.nb, cp.w);
                    }
                    if let Some(r) = rec.as_mut() {
                        r.point(cp.part, ph, &vb, &cp.xb, &cp.nb, cp.w);
                    }
                }
                for (e, &up) in wl.iter().zip(&under) {
                    if g.body[e.j] != bi as i32 || up >= 0 {
                        continue;
                    }
                    let (pI, _) = incident(g, pl, fc, pl.heads[h], &g.pg[e.j].c);
                    let mut ph = if pl.total { phi[(nrad + h) * Nf + e.j] } else { phi[(nrad + h) * Nf + e.j] + pI };
                    for r in 0..nrad {
                        ph += iw * Xr[r] * phi[r * Nf + e.j];
                    }
                    cdr.waterline(ph, &ax.pt(&e.x), &ax.vec(&g.pg[e.j].n), e.dl);
                    if let Some(r) = rec.as_mut() {
                        r.waterline(ph, &ax.pt(&e.x), &ax.vec(&g.pg[e.j].n), e.dl);
                    }
                }
                // the hydrostatic part by the closed forms of theory §11.2 over the wetted panels and
                // the plates (their area in the waterplane, no volume: z = 0)
                let mut hs = PressureDrift::new(&X, &b.rotation_center, rho, gr, w, true);
                for j in 0..Nf {
                    if g.mirror[j] == 0
                        && g.body[j] == bi as i32
                        && (g.kind[j] == SurfaceKind::Wet || g.kind[j] == SurfaceKind::Plate)
                    {
                        hs.hydrostatic_point(&ax.pt(&g.pg[j].c), &ax.vec(&g.pg[j].n), g.pg[j].c[2], g.pg[j].area, true);
                        if let Some(r) = rec.as_mut() {
                            r.hydrostatic_point(&ax.pt(&g.pg[j].c), &ax.vec(&g.pg[j].n), g.pg[j].c[2], g.pg[j].area);
                        }
                    }
                }
                cdr.add(&hs.finish_about_o(b.position[2], &[0.0f64; 36]));
                // the first-order force on the body (exciting, radiation, hydrostatic) for the
                // moment about the moving rotation centre
                let hy = m.hydrostatics(bi as i32);
                let off = m.derived()[bi].coordinate_offset as usize;
                let mut F1 = [CZ; 3];
                for d in 0..3 {
                    let x = out.X[h * D + off + d];
                    F1[d] = Complex64::new(x.re as f64, x.im as f64);
                    for r in 0..nrad {
                        let o = (off + d) * nrad + r;
                        F1[d] += (w * w * out.A[o] as f64 - iw * out.B[o] as f64) * Xr[r];
                    }
                    for k in 0..6 {
                        F1[d] -= hy.c_rho_g[d * 6 + k] * X[k];
                    }
                }
                cdr.finish(&F1, nine(&mut out.CD, (h * NB + bi) * 9));
            }
        }
        if let (Some(r), Some(s)) = (rec, out.qtf.first_mut()) {
            r.finish(s);
        }
    }
}

/// The source-formulation outputs of one ordinary frequency fi (out: its outputs; the parallel
/// loops on the current pool, work: their cancel check).
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn source_outputs<R: Real>(
    m: &Problem,
    g: &Geometry,
    pl: &Plan,
    sf: &Static<R>,
    fc: &FreqCtx,
    fi: i32,
    work: Work<'_>,
    in_: &SourceInput<R>,
    prof: &mut Profile,
    out: &mut Output,
) -> Result<()> {
    let mut s = Sources::new(m, g, pl, sf, fc, fi, work, in_, prof, out);
    s.source_strengths()?;
    if pl.fvel {
        s.field_velocities()?;
    }
    if (pl.pdrift && s.nh > 0) || pl.bvel {
        s.panel_velocities();
        if pl.bvel {
            s.body_velocities();
        }
        if pl.pdrift && s.nh > 0 {
            s.pressure_drift();
        }
    }
    if pl.cdrift && s.nh > 0 && in_.scs.is_some() {
        s.control_drift();
    }
    Ok(())
}
