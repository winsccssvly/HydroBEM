//! One frequency or limit of the low-order method: character matrices,
//! right-hand sides, LU, forces, Haskind, lid and pressure-surface elevations, field points, Kochin
//! functions, motions; the source-formulation outputs are in lo_velocity.rs. The kernels of a row
//! are evaluated four pairs at a time where they share a table tile (theory §8.3: the kernel is
//! most of the assembly, and only the vector width shortens it), the systems solved in complex64
//! (theory §13.1, grounds).

use crate::codes::Stage;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use num_complex::{Complex, Complex32, Complex64};
use rayon::prelude::*;

use crate::constants::PI;
use crate::error::Result;
use crate::geometry::{Vec3, v3};
use crate::green_fin::{FinPoint, FreqMode, Kern, fin_point};
use crate::lo_static::static_row;
use crate::lo_velocity::{source_outputs, to_cf};
use crate::low_order::{
    Column, FreqCtx, Geometry, Output, Plan, Profile, Real, SourceInput, Static, StaticEntry, StaticView, SurfaceKind,
    apply_kernel, channel_far_sheet, channel_grad, channel_influence, far_kernel, incident, influence, kochin_global,
    lid_column, pair_kernel, pair_kernels, plate_column, press_column, press_shape, sheet_wave, sheet_wave_fd,
};
use crate::motion::solve_motions;
use crate::parallel::{Block, Entries, in_batches, par_rows, try_for_each_grouped, write_blocks, write_entries};
use crate::problem::Problem;
use crate::progress::Work;
use crate::tanks::add_tank_terms;

const CZ: Complex64 = Complex64::new(0.0, 0.0);
// a tank's free surface is covered when its plates' area differs from it by less than this
// fraction (the rounding of the areas; a deck over part of a tank leaves a panel's area open)
const COVER_TOL: f64 = 1e-6;

// ------------------------------------------------------------------ one frequency

// One frequency or limit of the low-order method. The steps of solve_frequency (below) are its
// member functions, in their order; they share the sizes, the incident wave and boundary values,
// the character matrices and the solved potentials.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct Frequency<'a, R: Real> {
    g: &'a Geometry,
    pl: &'a Plan,
    st: &'a Static<R>,
    sf: &'a Static<R>,
    fc: &'a FreqCtx,
    work: Work<'a>,
    out: &'a mut Output,
    limit: bool,
    w: f64,
    Ns: i32,
    Nf: i32,
    G: i32,
    NC: i32,
    nrad: i32,
    nh: i32,
    nrhs: i32,
    N2: usize,
    D: i32,
    rho: f64,
    iwr: Complex64,
    inner: i32, // threads inside the frequency: the characters solved at once
    prof: Profile,
    phiI: Vec<Complex64>, // [nh][Nf] the incident wave; L phi_I on pressure surfaces
    dphiI: Vec<Complex64>,
    LphiI: Vec<Complex64>,
    u: Vec<Complex64>,     // [nrhs][Nf] the boundary values of the right-hand sides that use S
    phi: Vec<Complex64>,   // [nrhs][Nf] the solved potentials
    fpt: Vec<FinPoint>,    // finite depth: the one-point factors at every panel depth
    Smat: Vec<Complex<R>>, // [character] the source and dipole matrices, Ns x Ns column-major
    Amat: Vec<Complex<R>>,
    srcv: bool,                // the source formulation
    Vst: Vec<[Complex<R>; 3]>, // grad of the source integrals at the sector centroids
    // the direct method's extra columns of this frequency: their boundary values on every panel,
    // their free-surface terms E and their solutions [nx][Nf]
    ex: Option<&'a crate::problem::DirectFreq>,
    piv: Vec<Vec<usize>>, // [character] the row swaps of the factors kept in Amat (the direct method)
    ex_raw: Vec<Complex64>, // [nx][Nf] the direct method's sums over the field points, before / A_k
    ux: Vec<Complex64>,
    ex_e: Vec<Complex64>,
    interior: bool, // a tank problem (Problem::interior_domain)
}

// the free term of the row of panel j: -4 pi on the interior free surface (Lee, Newman & Zhu
// 1996 eq. 25), 4 pi on a lid, a free-surface plate and a pressure surface in z = 0 (a point of
// the free surface, where the source and its image coincide), 2 pi on the other panels
fn free_term(g: &Geometry, j: i32) -> f64 {
    let k = g.kind[j as usize];
    if k == SurfaceKind::Interior {
        -4.0 * PI
    } else if k == SurfaceKind::Lid || k == SurfaceKind::Plate || g.press_z0(j) {
        4.0 * PI
    } else {
        2.0 * PI
    }
}

// an unknown of panel j that is not used (an identity row): at the limits those of the interior
// free surface and of lids, at omega = inf also those of pressure surfaces in z = 0 (phi = 0)
fn unused_unknown(g: &Geometry, fc: &FreqCtx, j: i32, limit: bool) -> bool {
    let k = g.kind[j as usize];
    limit && (k == SurfaceKind::Interior || k == SurfaceKind::Lid || (g.press_z0(j) && fc.mode == FreqMode::Infinite))
}

/// A character's part of `solve`: its index, right-hand sides, block of Amat and pivots.
type SolveItem<'s, R> = (usize, &'s mut Vec<Complex<R>>, &'s mut [Complex<R>], &'s mut Vec<usize>);

impl<'a, R: Real> Frequency<'a, R> {
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    fn new(
        m: &'a Problem,
        g: &'a Geometry,
        pl: &'a Plan,
        st: &'a Static<R>,
        sf: &'a Static<R>,
        fc: &'a FreqCtx,
        work: Work<'a>,
        out: &'a mut Output,
    ) -> Self {
        let limit = fc.mode != FreqMode::Ordinary;
        let w = fc.w;
        let nrad = pl.rad.len() as i32;
        let nh = if pl.diffraction && !limit { pl.heads.len() as i32 } else { 0 };
        Frequency {
            g,
            pl,
            st,
            sf,
            fc,
            work,
            out,
            limit,
            w,
            Ns: g.Ns,
            Nf: g.Nf,
            G: g.G,
            NC: g.chi.len() as i32,
            nrad,
            nh,
            nrhs: nrad + nh,
            N2: g.Ns as usize * g.Ns as usize,
            D: pl.D,
            rho: pl.rho,
            iwr: Complex64::new(0.0, w * pl.rho),
            inner: m.opt.inner_threads,
            prof: Profile::new(),
            phiI: Vec::new(),
            dphiI: Vec::new(),
            LphiI: Vec::new(),
            u: Vec::new(),
            phi: Vec::new(),
            fpt: Vec::new(),
            Smat: Vec::new(),
            Amat: Vec::new(),
            srcv: false,
            Vst: Vec::new(),
            ex: None,
            piv: Vec::new(),
            ex_raw: Vec::new(),
            ux: Vec::new(),
            ex_e: Vec::new(),
            interior: m.interior_domain,
        }
    }

    /// The weights that pin the constant of a closed tank: in a tank problem whose free surface is
    /// rigid (omega = 0) or covered by plates (every frequency; at omega = infinity the constant of
    /// the odd extension, opposite in the image), a constant potential solves the homogeneous
    /// equation (theory §10.2), so its fully symmetric block is singular. Per sector unknown its
    /// area over the area of the potential unknowns (the walls and the plates; 0 on sheets and
    /// dampers, whose unknown is a jump); None where the problem is regular.
    fn constant_weights(&self) -> Option<Vec<f64>> {
        let g = self.g;
        if !self.interior {
            return None;
        }
        let potential = |j: usize| matches!(g.kind[j], SurfaceKind::Wet | SurfaceKind::Plate) && g.damper[j] < 0;
        if self.fc.mode != FreqMode::Zero {
            // covered: the plates take the whole free surface, whose area is that of the walls'
            // projection, sum n_z dS with the normals into the fluid
            let (mut plates, mut top) = (0.0, 0.0);
            for (j, p) in g.pg.iter().enumerate().take(g.Nf as usize) {
                if potential(j) && g.mirror[j] == 0 {
                    if g.kind[j] == SurfaceKind::Plate { plates += p.area } else { top += p.n[2] * p.area }
                }
            }
            if !(plates > 0.0 && (plates - top).abs() <= COVER_TOL * top.abs().max(plates)) {
                return None;
            }
        }
        let mut w: Vec<f64> =
            g.sec.iter().map(|&j| if potential(j as usize) { g.pg[j as usize].area } else { 0.0 }).collect();
        let total: f64 = w.iter().sum();
        if total <= 0.0 {
            return None;
        }
        w.iter_mut().for_each(|v| *v /= total);
        Some(w)
    }

    // generalized forces on a pressure coordinate: A - iB/w = rho int n_i phi_j dS,
    // X = -i w rho int n_i phi_D dS over its surface (theory §9.3)
    fn pweight(&self, ci: &Column, j: usize) -> f64 {
        let g = self.g;
        if g.mirror[j] == 0 && g.psurf[j] == ci.ps { press_shape(ci.shape, &g.lc[j]) * g.pg[j].area } else { 0.0 }
    }
    // the normal velocity n_i of a rigid or generalized coordinate on panel j
    fn ncoord(&self, ci: &Column, j: usize) -> f64 {
        if ci.mode > 0 { self.g.nj[j][(ci.mode - 1) as usize] } else { self.g.gn[ci.gm as usize][j] }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn boundary_values(&mut self) {
        let (g, pl, fc) = (self.g, self.pl, self.fc);
        let (nh, Nf, nrad, nrhs) = (self.nh as usize, self.Nf as usize, self.nrad as usize, self.nrhs as usize);
        // incident wave on all panels
        // (and L phi_I = dphi_I/dz - K phi_I on pressure surfaces, theory §9.3)
        self.phiI = vec![CZ; nh * Nf];
        self.dphiI = vec![CZ; nh * Nf];
        if !g.ps.is_empty() {
            self.LphiI = vec![CZ; nh * Nf];
        }
        for h in 0..nh {
            for j in 0..Nf {
                let (pI, gr) = incident(g, pl, fc, pl.heads[h], &g.pg[j].c);
                self.phiI[h * Nf + j] = pI;
                self.dphiI[h * Nf + j] = gr[0] * g.pg[j].n[0] + gr[1] * g.pg[j].n[1] + gr[2] * g.pg[j].n[2];
                if g.kind[j] == SurfaceKind::Press {
                    self.LphiI[h * Nf + j] = gr[2] - fc.K * pI;
                }
            }
        }
        // boundary values u (normal velocity, Nf) of each right-hand side that uses S
        self.u = vec![CZ; nrhs * Nf];
        // (a pressure mode: L phi_j = n_j on its surface, so dphi_j/dn = -K phi_j - n_j
        // with the K term in the column)
        for c in 0..nrad {
            for j in 0..Nf {
                let col = &pl.rad[c];
                // an impervious damper: psi_n = phi_n, no body velocity on it
                let imperv = g.damper[j] >= 0 && !g.dmp[g.damper[j] as usize].porous;
                if col.mode > 0 && g.body[j] == col.body && !imperv {
                    self.u[c * Nf + j] = Complex64::new(g.nj[j][(col.mode - 1) as usize], 0.0);
                }
                if col.mode < 0 && g.body[j] == col.body && !imperv {
                    self.u[c * Nf + j] = Complex64::new(g.gn[col.gm as usize][j], 0.0);
                }
                if col.mode == 0 && g.psurf[j] == col.ps {
                    self.u[c * Nf + j] = Complex64::new(-press_shape(col.shape, &g.lc[j]), 0.0);
                }
            }
        }
        if !pl.total {
            for h in 0..nh {
                for j in 0..Nf {
                    if g.body_surface(j as i32) {
                        self.u[(nrad + h) * Nf + j] = -self.dphiI[h * Nf + j];
                    }
                    if g.kind[j] == SurfaceKind::Press {
                        self.u[(nrad + h) * Nf + j] = self.LphiI[h * Nf + j]; // L phi_S = -L phi_I
                    }
                    if g.kind[j] == SurfaceKind::Lid {
                        // the condition of the total potential on a lid, phi_Sz - kappa phi_S =
                        // (kappa - K) phi_I: the kappa term in the column, the rest the boundary
                        // value along the normal into the fluid (-z), (K - kappa) phi_I
                        self.u[(nrad + h) * Nf + j] = (fc.K - fc.kappa[g.lid[j] as usize]) * self.phiI[h * Nf + j];
                    }
                }
            }
        }
        self.phi = vec![CZ; nrhs * Nf];
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn matrices(&mut self) -> Result<()> {
        let (g, pl, fc, limit) = (self.g, self.pl, self.fc, self.limit);
        let (Nsu, Nfu, NCu, N2) = (self.Ns as usize, self.Nf as usize, self.NC as usize, self.N2);
        // character matrices S_c and D_c (column-major, Ns x Ns each); every pair
        // kernel is evaluated once for all characters and for both (k, g n) and
        // (n, g k), which share it: G(gx, g xi) = G(x, xi) and G(x, xi) = G(xi, x)
        // finite depth: the one-point factors of the kernel at every panel depth
        if fc.finite() {
            self.fpt = (0..Nfu).map(|j| fin_point(&fc.fd, g.pg[j].c[2])).collect();
        }
        let mut Smat = vec![Complex::<R>::default(); NCu * N2];
        let mut Amat = vec![Complex::<R>::default(); NCu * N2];
        let zero = Complex::<R>::default();
        {
            // the items below compute blocks of entries of the matrices S_c (matrix c) and A_c
            // (matrix NC + c), written into them after each batch of items
            let mut mats: Vec<&mut [Complex<R>]> =
                Smat.chunks_mut(N2.max(1)).chain(Amat.chunks_mut(N2.max(1))).collect();
            if g.W > 0.0 {
                self.assemble_channel_rows(&mut mats)?;
            }
            // source formulation: grad_x of the source integral of every panel j at
            // every sector centroid k, from the same pair kernels (the field derivatives for
            // (k, g n), the source derivatives reflected for (n, g k))
            self.srcv = g.source && !limit && (((pl.pdrift || pl.cdrift) && self.nh > 0) || pl.fvel || pl.bvel);
            let mut Vst = vec![[zero; 3]; if self.srcv { Nsu * Nfu } else { 0 }];
            if self.srcv && g.W > 0.0 {
                self.channel_gradients(&mut Vst)?;
            }
            self.assemble_tiles(&mut mats, &mut Vst)?;
            self.Vst = Vst;
        }
        self.Smat = Smat;
        self.Amat = Amat;
        self.prof.mark("  assembly (all characters)");
        Ok(())
    }

    // channel: every (k, n) pair of the matrices with the image and tail sums; the rows on the
    // threads
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn assemble_channel_rows(&self, mats: &mut [&mut [Complex<R>]]) -> Result<()> {
        let (g, st, fc, work, fpt) = (self.g, self.st, self.fc, self.work, &self.fpt);
        let (Nsu, Nfu, G, NCu, limit) = (self.Ns as usize, self.Nf as usize, self.G, self.NC as usize, self.limit);
        let zero = Complex::<R>::default();
        let one = Complex::new(R::from_f64(1.0), R::from_f64(0.0));
        let threads = rayon::current_num_threads();
        let rows: Vec<usize> = (0..Nsu).collect();
        let row = |&ku: &usize| -> Result<Block<Complex<R>>> {
            work.check_cancel()?;
            let mut b = Block::new(ku, 1, 0, Nsu, 2 * NCu, zero);
            let rowk = ku * Nfu;
            let sk = g.sec[ku];
            let xk = &g.pg[sk as usize].c;
            // the rows of unused unknowns are identity rows as in the open sea
            // (assemble_tiles); here their columns are zeroed as well, which the open-sea
            // tiles do not do (the same solution: those unknowns have identity rows and a
            // zero right-hand side)
            let ik = unused_unknown(g, fc, sk, limit);
            let free_k = free_term(g, sk);
            for n in 0..Nsu {
                let mut skn = [CZ; 4];
                let mut dkn = [CZ; 4];
                for gi in 0..G as usize {
                    let j = g.img[gi * Nsu + n];
                    let (S1, D1) = channel_influence(g, st, rowk, j, xk, &fpt[sk as usize], &fpt[j as usize], fc);
                    for ch in 0..NCu {
                        skn[ch] += g.chi[ch][gi] * S1;
                        dkn[ch] += g.chi[ch][gi] * D1;
                    }
                }
                let in_ = unused_unknown(g, fc, g.sec[n], limit);
                for ch in 0..NCu {
                    // row k of every character is computed by item k only
                    if ik {
                        b.set(ch, ku, n, zero);
                        b.set(NCu + ch, ku, n, if ku == n { one } else { zero });
                    } else if in_ {
                        b.set(ch, ku, n, zero);
                        b.set(NCu + ch, ku, n, zero);
                    } else {
                        b.set(ch, ku, n, R::c_from(skn[ch]));
                        b.set(
                            NCu + ch,
                            ku,
                            n,
                            R::c_from((if ku == n { Complex64::new(free_k, 0.0) } else { CZ }) - dkn[ch]),
                        );
                    }
                }
            }
            Ok(b)
        };
        in_batches(&rows, 16 * threads, row, |blocks| {
            write_blocks(mats, Nsu, &blocks.iter().collect::<Vec<_>>());
        })
    }

    // channel, source formulation: the gradients Vst directly, every sector row and panel
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn channel_gradients(&self, Vst: &mut [[Complex<R>; 3]]) -> Result<()> {
        let (g, st, fc, work, fpt) = (self.g, self.st, self.fc, self.work, &self.fpt);
        let (Nsu, Nf, Nfu) = (self.Ns as usize, self.Nf, self.Nf as usize);
        let vg: [&[[R; 3]]; 3] = [&st.vg, &st.vg1, &st.vg2];
        let vl: [&[[R; 3]]; 3] = [&st.vl, &st.vl1, &st.vl2];
        // row k of Vst is written by item k only
        par_rows(Vst, Nfu, Nsu).enumerate().try_for_each(|(ku, row)| -> Result<()> {
            work.check_cancel()?;
            let sk = g.sec[ku] as usize;
            for j in 0..Nf {
                let o = ku * Nfu + j as usize;
                let v = channel_grad(g, vg, vl, o, j, &g.pg[sk].c, &fpt[sk], &fpt[j as usize], fc);
                for d in 0..3 {
                    row[j as usize][d] = R::c_from(v[d]);
                }
            }
            Ok(())
        })
    }

    // the open sea: the matrices and the gradients Vst of the source formulation (Vst empty
    // without it) by tile pairs; nothing in a channel (no tiles)
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn assemble_tiles(&self, mats: &mut [&mut [Complex<R>]], Vst: &mut [[Complex<R>; 3]]) -> Result<()> {
        let (g, st, fc, work, fpt) = (self.g, self.st, self.fc, self.work, &self.fpt);
        let (Ns, G, NCu, limit, srcv) = (self.Ns, self.G, self.NC as usize, self.limit, self.srcv);
        let (Nsu, Nfu) = (Ns as usize, self.Nf as usize);
        let zero = Complex::<R>::default();
        let one = Complex::new(R::from_f64(1.0), R::from_f64(0.0));
        let threads = rayon::current_num_threads();
        let c2v = 2.0 * fc.K * fc.logc;
        // the gradient (k, j) at its index in Vst
        let vgrad = |k: usize, j: usize, wr: &[f64; 3], wi: &[f64; 3]| -> (usize, [Complex<R>; 3]) {
            let o = k * Nfu + j;
            let A = g.pg[j].area;
            let mut v = [zero; 3];
            for d in 0..3 {
                v[d] =
                    R::c_from(Complex64::new(st.vg[o][d].as_f64() + A * wr[d] - c2v * st.vl[o][d].as_f64(), A * wi[d]));
            }
            (o, v)
        };
        // the (k, n) loops in tiles (cache blocking); the tile pairs (k0 <= n0) on the threads: a pair
        // computes the entries (k, n) and (n, k) of its tiles and the gradients Vst(k, g n) and
        // Vst(n, g k), which no other pair computes, so the matrices do not depend on the thread count
        const TILE: i32 = 48;
        let mut tiles: Vec<(i32, i32)> = Vec::new();
        let mut k0 = 0;
        while k0 < Ns && g.W == 0.0 {
            let mut n0 = k0;
            while n0 < Ns {
                tiles.push((k0, n0));
                n0 += TILE;
            }
            k0 += TILE;
        }
        let tiles_done = AtomicI32::new(0); // the progress
        let ntiles = tiles.len() as i32;
        // the entries of a tile pair: the block of the entries (k, n), that of the entries (n, k)
        // (the same block for a diagonal pair), and the gradients in the rows k and n of Vst
        #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
        struct TileOut<C> {
            blocks: Vec<Block<C>>,
            grads: [Entries<[C; 3]>; 2],
        }
        let tile = |&(k0, n0): &(i32, i32)| -> Result<TileOut<Complex<R>>> {
            work.check_cancel()?;
            let (k1, n1) = (Ns.min(k0 + TILE) as usize, Ns.min(n0 + TILE) as usize);
            let (k0u, n0u) = (k0 as usize, n0 as usize);
            let mut kn = Block::new(k0u, k1 - k0u, n0u, n1 - n0u, 2 * NCu, zero);
            let mut nk = if k0 == n0 { None } else { Some(Block::new(n0u, n1 - n0u, k0u, k1 - k0u, 2 * NCu, zero)) };
            let mut gk = Entries { band: k0u / TILE as usize, e: Vec::new() };
            let mut gn = Entries { band: n0u / TILE as usize, e: Vec::new() };
            let mut js: Vec<i32> = Vec::with_capacity(TILE as usize * G as usize);
            let mut kerns: Vec<(Kern, f64)> = Vec::with_capacity(TILE as usize * G as usize);
            for k in k0..Ns.min(k0 + TILE) {
                let ku = k as usize;
                let rowk = ku * Nfu;
                let xk = &g.pg[g.sec[ku] as usize].c;
                // the kernels of the row's pairs in the order of the loops below (n, then the
                // image gi), four at a time where they apply
                js.clear();
                kerns.clear();
                for n in k.max(n0)..Ns.min(n0 + TILE) {
                    js.extend((0..G as usize).map(|gi| g.img[gi * Nsu + n as usize]));
                }
                let px = if fc.finite() { Some(&fpt[g.sec[ku] as usize]) } else { None };
                pair_kernels(fc, g, xk, px, fpt, &js, &mut kerns);
                let mut kit = kerns.iter();
                for n in k.max(n0)..Ns.min(n0 + TILE) {
                    let nu = n as usize;
                    let rown = nu * Nfu;
                    let xn = &g.pg[g.sec[nu] as usize].c;
                    let mut skn = [CZ; 4];
                    let mut dkn = [CZ; 4];
                    let mut snk = [CZ; 4];
                    let mut dnk = [CZ; 4];
                    for gi in 0..G as usize {
                        let j = g.img[gi * Nsu + nu];
                        // the kernel of the pair and its distance
                        let &(wk, Rk) = kit.next().unwrap();
                        let (S1, D1) = apply_kernel(g, st, rowk, j, xk, fc, &wk, Rk, false);
                        if srcv {
                            let y = &g.pg[j as usize].c;
                            let ax = if Rk > 0.0 { (xk[0] - y[0]) / Rk } else { 0.0 };
                            let ay = if Rk > 0.0 { (xk[1] - y[1]) / Rk } else { 0.0 };
                            let wr = [wk.WrR * ax, wk.WrR * ay, wk.Wrz];
                            let wi = [wk.WiR * ax, wk.WiR * ay, wk.Wiz];
                            gk.e.push(vgrad(ku, j as usize, &wr, &wi));
                            if n != k {
                                // at x_n of panel g k: R_g grad_y G(x_k, y) at y = g x_n (the source derivatives)
                                let sx = g.refl[gi].sx;
                                let sy = g.refl[gi].sy;
                                let wr2 = [-sx * wk.WrR * ax, -sy * wk.WrR * ay, wk.WrZ];
                                let wi2 = [-sx * wk.WiR * ax, -sy * wk.WiR * ay, wk.WiZ];
                                gn.e.push(vgrad(nu, g.img[gi * Nsu + ku] as usize, &wr2, &wi2));
                            }
                        }
                        let (mut S2, mut D2) = (S1, D1);
                        if n != k {
                            (S2, D2) = apply_kernel(g, st, rown, g.img[gi * Nsu + ku], xn, fc, &wk, Rk, true);
                        }
                        for ch in 0..NCu {
                            let x = g.chi[ch][gi];
                            skn[ch] += x * S1;
                            dkn[ch] += x * D1;
                            snk[ch] += x * S2;
                            dnk[ch] += x * D2;
                        }
                    }
                    // the unused unknowns: identity rows
                    let ik = unused_unknown(g, fc, g.sec[ku], limit);
                    let in_ = unused_unknown(g, fc, g.sec[nu], limit);
                    let free_k = free_term(g, g.sec[ku]);
                    for ch in 0..NCu {
                        // the entries (k, n) and (n, k) are computed by this tile pair only
                        if ik {
                            kn.set(ch, ku, nu, zero);
                            kn.set(NCu + ch, ku, nu, if k == n { one } else { zero });
                        } else {
                            let free = if k == n { Complex64::new(free_k, 0.0) } else { CZ };
                            kn.set(ch, ku, nu, R::c_from(skn[ch]));
                            kn.set(NCu + ch, ku, nu, R::c_from(free - dkn[ch]));
                        }
                        if n != k {
                            let b = nk.as_mut().unwrap_or(&mut kn);
                            b.set(ch, nu, ku, if in_ { zero } else { R::c_from(snk[ch]) });
                            b.set(NCu + ch, nu, ku, if in_ { zero } else { R::c_from(-dnk[ch]) });
                        }
                    }
                }
            }
            work.report(Stage::Assembly, tiles_done.fetch_add(1, Ordering::SeqCst) + 1, ntiles);
            Ok(TileOut { blocks: std::iter::once(kn).chain(nk).collect(), grads: [gk, gn] })
        };
        in_batches(&tiles, 4 * threads, tile, |outs| {
            write_blocks(mats, Nsu, &outs.iter().flat_map(|o| o.blocks.iter()).collect::<Vec<_>>());
            if srcv {
                write_entries(Vst, TILE as usize * Nfu, &outs.iter().flat_map(|o| o.grads.iter()).collect::<Vec<_>>());
            }
        })
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn sheet_rows(&mut self) -> Result<()> {
        let (g, st, fc) = (self.g, self.st, self.fc);
        let (Ns, Nf, G, NC, N2, w, work) = (self.Ns, self.Nf, self.G, self.NC, self.N2, self.w, self.work);
        let (Nsu, Nfu, NCu) = (Ns as usize, Nf as usize, NC as usize);
        // thin sheets: no source on a sheet column; a sheet row
        // is the normal derivative of Green's identity,
        //   sum_n H(k, n) x_n = 4 pi u_k + sum_{n wet} K'(k, n) u_n  (-4 pi dphi_I/dn, total),
        // H = int d2G/dn_x dn_xi, K' = int dG/dn_x
        // At omega = inf a free-surface plate is a sheet in z = 0 (theory §9.8): with the odd
        // image of phi = 0 on the free surface its unknown phi (the water side) is half the jump,
        // its coincident image doubles the dipole and the hypersingular kernel of its columns and
        // rows (those of G = 1/r - 1/r'), and its single layer vanishes
        if g.sheet_rows(fc) {
            let zero = Complex::<R>::default();
            let N2c = N2;
            for n in 0..Nsu {
                if g.sheet_like(fc, g.sec[n]) {
                    for ch in 0..NC as usize {
                        for r in 0..Nsu {
                            self.Smat[ch * N2c + r + n * Nsu] = zero;
                        }
                    }
                }
            }
            let c2 = 2.0 * fc.K * fc.logc;
            let fpt = &self.fpt;
            // the sheet rows on the threads, each row k of every character a block of its own
            let rows: Vec<usize> = (0..Nsu).filter(|&k| st.sheet[k] >= 0 && g.sheet_like(fc, g.sec[k])).collect();
            let row = |&ku: &usize| -> Result<Block<Complex<R>>> {
                work.check_cancel()?;
                let mut b = Block::new(ku, 1, 0, Nsu, 2 * NCu, zero);
                let qu = st.sheet[ku] as usize;
                let sk = g.sec[ku] as usize;
                let pk = &g.pg[sk];
                for n in 0..Nsu {
                    let mut h = [CZ; 4];
                    let mut kp = [CZ; 4];
                    for gi in 0..G as usize {
                        let j = g.img[gi * Nsu + n] as usize;
                        let e = &st.sh[qu * Nfu + j];
                        let inf = fc.mode == FreqMode::Infinite;
                        let (Hw, Kw) = if fc.finite() {
                            sheet_wave_fd(fc, &pk.c, &pk.n, &g.pg[j], Some(&fpt[sk]), Some(&fpt[j]))
                        } else {
                            sheet_wave(fc, &pk.c, &pk.n, &g.pg[j])
                        };
                        let mut H = (if inf { st.shinf[qu * Nfu + j][0].as_f64() } else { e[0].as_f64() }) + Hw
                            - c2 * e[2].as_f64();
                        let mut Kq = (if inf { st.shinf[qu * Nfu + j][1].as_f64() } else { e[1].as_f64() }) + Kw
                            - c2 * e[3].as_f64();
                        if g.W > 0.0 && g.Rc != 0.0 {
                            // channel: the wall images (the sheet point and its normal reflected,
                            // y -> +-W - y) and the far images, scaled by the reflection factor (none without it)
                            let nr = v3(pk.n[0], -pk.n[1], pk.n[2]);
                            let (sh, si) = ([&st.sh1, &st.sh2], [&st.shinf1, &st.shinf2]);
                            for (m, ym) in [g.W - pk.c[1], -g.W - pk.c[1]].into_iter().enumerate() {
                                let xm = v3(pk.c[0], ym, pk.c[2]);
                                let e = &sh[m][qu * Nfu + j];
                                let (Hw, Kw) = sheet_wave_fd(fc, &xm, &nr, &g.pg[j], Some(&fpt[sk]), Some(&fpt[j]));
                                let o = qu * Nfu + j;
                                H += g.Rc
                                    * ((if inf { si[m][o][0].as_f64() } else { e[0].as_f64() }) + Hw
                                        - c2 * e[2].as_f64());
                                Kq += g.Rc
                                    * ((if inf { si[m][o][1].as_f64() } else { e[1].as_f64() }) + Kw
                                        - c2 * e[3].as_f64());
                            }
                            let p = &g.pg[j];
                            let (Hf, Kf) =
                                channel_far_sheet(fc, g.W, &pk.c, &pk.n, &p.c, &p.n, p.area, &fpt[sk], &fpt[j]);
                            H += g.Rc * Hf;
                            Kq += g.Rc * Kf;
                        }
                        for ch in 0..NCu {
                            h[ch] += g.chi[ch][gi] * H;
                            kp[ch] += g.chi[ch][gi] * Kq;
                        }
                    }
                    let sn = g.sec[n];
                    let sheet_col = g.sheet_like(fc, sn);
                    // the normal derivative of the column of each kind (the dipole column D of
                    // the body rows): a lid -(K - kappa) S, a plate -K S (at omega = inf the
                    // plate's H, a sheet's), a pressure surface D + K S (0 in z = 0); not
                    // kind_column: an interior column keeps the normal derivative of D here
                    if g.kind[sn as usize] == SurfaceKind::Lid {
                        for ch in 0..NCu {
                            h[ch] = lid_column(g, fc, sn, kp[ch]); // unused at the limits
                        }
                    } else if g.kind[sn as usize] == SurfaceKind::Plate && !sheet_col {
                        for ch in 0..NCu {
                            h[ch] = plate_column(fc, kp[ch]);
                        }
                    } else if g.kind[sn as usize] == SurfaceKind::Press {
                        for ch in 0..NCu {
                            h[ch] = press_column(g, fc, sn, kp[ch], h[ch]);
                        }
                    }
                    for ch in 0..NCu {
                        b.set(NCu + ch, ku, n, R::c_from(h[ch]));
                        b.set(
                            ch,
                            ku,
                            n,
                            if sheet_col {
                                if ku == n { Complex::new(R::from_f64(-4.0 * PI), R::from_f64(0.0)) } else { zero }
                            } else {
                                R::c_from(-kp[ch])
                            },
                        );
                    }
                }
                Ok(b)
            };
            {
                let mut mats: Vec<&mut [Complex<R>]> =
                    self.Smat.chunks_mut(N2.max(1)).chain(self.Amat.chunks_mut(N2.max(1))).collect();
                in_batches(&rows, 16 * rayon::current_num_threads(), row, |blocks| {
                    write_blocks(&mut mats, Nsu, &blocks.iter().collect::<Vec<_>>());
                })?;
            }
            // dipole dampers (theory §9.5): the unknown on a
            // damper is psi_n = phi_n - v_n with the jump mu = c psi_n, c = -i eps / w (the
            // jump mu = phi(+n) - phi(-n) of the sign convention of theory §9.5), so
            // its column is scaled by c and its row gains -4 pi psi_n (u on it is v_n)
            for n in 0..Nsu {
                let d = g.damper[g.sec[n] as usize];
                if d < 0 {
                    continue;
                }
                let cn = Complex64::new(0.0, -g.dmp[d as usize].eps / w);
                for ch in 0..NC as usize {
                    let col = &mut self.Amat[ch * N2 + n * Nsu..ch * N2 + (n + 1) * Nsu];
                    for x in col.iter_mut() {
                        *x = R::c_from(cn * R::c_as_cd(*x));
                    }
                    col[n] = R::c_from(R::c_as_cd(col[n]) - 4.0 * PI);
                }
            }
            self.prof.mark("  assembly (sheet rows)");
        }
        Ok(())
    }

    /// The direct method's extra columns: their boundary values on every panel (from the body
    /// points) and the gradient part of their free-surface terms, Σ_q wg_q · ∇_h S_qk over the points
    /// with velocity rows, into ex_raw (the sums of ws_q S_qk come with the field points' pass,
    /// field_values; E_k = ex_raw_k / A_k plus the body terms, solve_extra).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn direct_gradients(&mut self, m: &Problem) -> Result<()> {
        let (g, pl, fc, sf) = (self.g, self.pl, self.fc, self.sf);
        let d = self.ex.expect("the extra columns");
        let (Nf, nx) = (g.Nf as usize, d.ncols);
        let Nb = pl.bp.len();
        self.ux = vec![CZ; nx * Nf];
        for c in 0..nx {
            for (q, &j) in pl.bp.iter().enumerate() {
                self.ux[c * Nf + j as usize] = d.u[c * Nb + q];
            }
        }
        self.ex_e = vec![CZ; nx * Nf];
        self.ex_raw = vec![CZ; nx * Nf];
        let P = g.fp.len();
        let vfrom = m.opt.field_velocity_from.min(P);
        let c2f = 2.0 * fc.K * fc.logc;
        let work = self.work;
        let wg = |q: usize| &d.wg[(q - vfrom) * nx..(q - vfrom + 1) * nx];
        let pts: Vec<usize> = (vfrom..P).filter(|&q| wg(q).iter().any(|w| w[0] != CZ || w[1] != CZ)).collect();
        for batch in pts.chunks(DIRECT_BATCH) {
            // the two gradient rows of every point of the batch on the threads, with their weights
            let rows: Vec<Vec<(Vec<Complex64>, Vec<Complex64>)>> = batch
                .par_iter()
                .map(|&q| -> Result<Vec<(Vec<Complex64>, Vec<Complex64>)>> {
                    work.check_cancel()?;
                    let x = &g.fp[q];
                    let (vg_own, vl_own) = match &sf.lazy {
                        Some(lz) => {
                            let mut a = vec![[R::default(); 3]; Nf];
                            let mut bl = vec![[R::default(); 3]; Nf];
                            crate::lo_static::source_row(g, x, lz.logint, lz.tol, lz.depth, &mut a, &mut bl);
                            (a, bl)
                        }
                        None => (Vec::new(), Vec::new()),
                    };
                    let mut gr = [vec![CZ; Nf], vec![CZ; Nf]];
                    for j in 0..Nf {
                        let A = g.pg[j].area;
                        let (wk, Rr) = pair_kernel(fc, x, &g.pg[j].c, None, None);
                        let ax = if Rr > 0.0 { (x[0] - g.pg[j].c[0]) / Rr } else { 0.0 };
                        let ay = if Rr > 0.0 { (x[1] - g.pg[j].c[1]) / Rr } else { 0.0 };
                        let wr = [wk.WrR * ax, wk.WrR * ay];
                        let wi = [wk.WiR * ax, wk.WiR * ay];
                        let (vgj, vlj) = if sf.lazy.is_some() {
                            (vg_own[j], vl_own[j])
                        } else {
                            let o = (q - sf.vfrom) * Nf + j;
                            (sf.vg[o], sf.vl[o])
                        };
                        for dd in 0..2 {
                            gr[dd][j] =
                                Complex64::new(vgj[dd].as_f64() + A * wr[dd] - c2f * vlj[dd].as_f64(), A * wi[dd]);
                        }
                    }
                    let [g0, g1] = gr;
                    let w = wg(q);
                    Ok(vec![(w.iter().map(|v| v[0]).collect(), g0), (w.iter().map(|v| v[1]).collect(), g1)])
                })
                .collect::<Result<Vec<_>>>()?;
            let rows: Vec<(Vec<Complex64>, Vec<Complex64>)> = rows.into_iter().flatten().collect();
            add_weighted_rows(&mut self.ex_raw, nx, Nf, &rows);
        }
        self.prof.mark("  direct: free-surface gradient terms");
        Ok(())
    }

    /// The direct method's Stokes terms of a floating body: E_k −= g Σ_q bt_q · V_qk / A_k over the
    /// body points q (E is divided by g in the right-hand side), V_qk = ∇ of the source integral over
    /// panel k at q, from the gradients Vst of the sector rows: V(img(γ, s), k) = R_γ V(s, γ k).
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn direct_body_terms(&mut self) -> Result<()> {
        let (g, pl) = (self.g, self.pl);
        let d = self.ex.expect("the extra columns");
        let (Ns, Nf, G, nx) = (g.Ns as usize, g.Nf as usize, g.G as usize, d.ncols);
        if self.Vst.len() != Ns * Nf {
            return Err(crate::Error::internal("the direct method of a floating body without the source gradients"));
        }
        // the panel γ k of every panel k
        let gk: Vec<usize> =
            g.panel_images().into_iter().map(|j| if j < 0 { usize::MAX } else { j as usize }).collect();
        // the body point of every panel, the weights in the geometry's axes
        let mut bq = vec![usize::MAX; Nf];
        for (q, &j) in pl.bp.iter().enumerate() {
            bq[j as usize] = q;
        }
        let (cy, sy) = ((g.yaw * PI / 180.0).cos(), (g.yaw * PI / 180.0).sin());
        let local = |v: &[Complex64; 3]| [cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2]];
        let bt: Vec<[Complex64; 3]> = d.bt.iter().map(local).collect();
        // the sector body rows s with the body point of each image
        let srows: Vec<usize> =
            (0..Ns).filter(|&s| (0..G).any(|gi| bq[g.img[gi * Ns + s] as usize] != usize::MAX)).collect();
        let (Vst, grav, work) = (&self.Vst, pl.grav, self.work);
        let terms: Vec<Vec<Complex64>> = (0..Nf)
            .into_par_iter()
            .map(|k| -> Result<Vec<Complex64>> {
                if k % 256 == 0 {
                    work.check_cancel()?;
                }
                let mut acc = vec![CZ; nx];
                for gi in 0..G {
                    let kk = gk[gi * Nf + k];
                    if kk == usize::MAX {
                        continue;
                    }
                    let (sx, syy) = (g.refl[gi].sx, g.refl[gi].sy);
                    for &s in &srows {
                        let q = bq[g.img[gi * Ns + s] as usize];
                        if q == usize::MAX {
                            continue;
                        }
                        let e = &Vst[s * Nf + kk];
                        let v = [sx * R::c_as_cd(e[0]), syy * R::c_as_cd(e[1]), R::c_as_cd(e[2])];
                        let w = &bt[q * nx..(q + 1) * nx];
                        for c in 0..nx {
                            acc[c] += w[c][0] * v[0] + w[c][1] * v[1] + w[c][2] * v[2];
                        }
                    }
                }
                Ok(acc)
            })
            .collect::<Result<Vec<_>>>()?;
        for k in 0..Nf {
            let a = g.pg[k].area;
            for c in 0..nx {
                self.ex_e[c * Nf + k] -= grav * terms[k][c] / a;
            }
        }
        self.prof.mark("  direct: body terms (Stokes)");
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn solve(&mut self) -> Result<()> {
        let (g, pl) = (self.g, self.pl);
        let (Ns, Nf, G, NC, N2, inner, nrad, nrhs, work) =
            (self.Ns, self.Nf, self.G, self.NC, self.N2, self.inner, self.nrad, self.nrhs, self.work);
        let (Nsu, Nfu, Gu) = (Ns as usize, Nf as usize, G as usize);
        // the characters on the threads, at most inner at once, each with its own right-hand sides
        // and matrix copy, the blocks of their LUs on the same threads; the potentials are
        // then summed over the characters in their order
        let mut sol: Vec<Vec<Complex<R>>> = vec![Vec::new(); NC as usize];
        let tot = nrhs as usize;
        // the direct method solves its extra columns later with the factors, kept in Amat (which
        // nothing reads after the solve: every solve factors it in place)
        let keep = self.ex.is_some() && !g.wavemaker;
        self.piv = vec![if keep { vec![0usize; Nsu] } else { Vec::new() }; NC as usize];
        let pin = self.constant_weights();
        {
            let (Smat, u, phiI, dphiI) = (&self.Smat, &self.u, &self.phiI, &self.dphiI);
            let mut items: Vec<SolveItem<'_, R>> = sol
                .iter_mut()
                .zip(self.Amat.chunks_mut(N2.max(1)))
                .zip(self.piv.iter_mut())
                .enumerate()
                .map(|(c, ((s, a), p))| (c, s, a, p))
                .collect();
            try_for_each_grouped(&mut items, inner.min(NC) as usize, |(chu, rhs, amat, piv)| {
                work.check_cancel()?;
                // item ch writes sol[ch], its Amat block and its pivots only
                let (chu, rhs) = (*chu, &mut **rhs);
                *rhs = vec![Complex::<R>::default(); Nsu * tot];
                let mut U = vec![Complex::<R>::default(); Nsu * tot];
                let chi = &g.chi[chu];
                let Sc = &Smat[chu * N2..(chu + 1) * N2];
                // right-hand sides projected on the character: 4 pi phi_I for the
                // total diffraction potential, -S u otherwise (one matrix product)
                let mut nS = 0usize;
                let mut colS: Vec<usize> = Vec::new();
                for c in 0..nrhs as usize {
                    if c >= nrad as usize && pl.total {
                        let h = c - nrad as usize;
                        for k in 0..Nsu {
                            let mut v = CZ;
                            // a sheet row: -4 pi dphi_I/dn; not reached while the rules refuse sheets
                            // with the total formulation
                            let sheet = g.kind[g.sec[k] as usize] == SurfaceKind::Sheet;
                            let f = if sheet { dphiI } else { phiI };
                            for gi in 0..Gu {
                                v += chi[gi] * f[h * Nfu + g.img[gi * Nsu + k] as usize];
                            }
                            rhs[k + c * Nsu] = R::c_from((if sheet { -4.0 } else { 4.0 }) * PI * v / G as f64);
                        }
                        continue;
                    }
                    for n in 0..Nsu {
                        let mut v = CZ;
                        for gi in 0..Gu {
                            v += chi[gi] * u[c * Nfu + g.img[gi * Nsu + n] as usize];
                        }
                        U[n + nS * Nsu] = R::c_from(-v / G as f64);
                    }
                    colS.push(c);
                    nS += 1;
                }
                if nS > 0 {
                    let mut prod = vec![Complex::<R>::default(); Nsu * nS];
                    // P = S U for the right-hand sides (column-major, S n x n, U n x m)
                    // Sc is Ns x Ns, U Ns x nS (or more), prod Ns x nS
                    R::gemm(Nsu, nS, Nsu, Sc, Nsu, &U, Nsu, &mut prod, Nsu);
                    for q in 0..nS {
                        rhs[colS[q] * Nsu..(colS[q] + 1) * Nsu].copy_from_slice(&prod[q * Nsu..(q + 1) * Nsu]);
                    }
                }
                // a closed tank: its fully symmetric block takes 4 pi w^T on every row of a
                // potential unknown, which pins the area-weighted mean of the potential to 0 and
                // changes no solution of the singular system (theory §10.2)
                if let Some(w) = &pin
                    && chi.iter().all(|&c| c > 0.0)
                {
                    for (k, &wk) in w.iter().enumerate().filter(|(_, wk)| **wk > 0.0) {
                        let add = R::c_from(Complex64::new(4.0 * PI * wk, 0.0));
                        for (i, &wi) in w.iter().enumerate() {
                            if wi > 0.0 {
                                amat[i + k * Nsu] = amat[i + k * Nsu] + add;
                            }
                        }
                    }
                }
                if keep {
                    R::lu_factor(Ns, amat, piv, work)?;
                }
                if tot == 0 {
                    return Ok(());
                }
                if g.wavemaker {
                    // wavemakers in the walls: every panel coincides with its wall image of the
                    // opposite normal, so the dipoles cancel and 4 pi phi = -int G u over all images
                    let s = R::from_f64(4.0 * PI);
                    for v in rhs.iter_mut() {
                        *v = *v / s;
                    }
                } else if keep {
                    R::lu_substitute(Ns, amat, piv, tot as i32, rhs, work)?;
                } else {
                    R::lu_solve(Ns, amat, tot as i32, rhs, work)?;
                }
                Ok(())
            })?;
        }
        self.prof.mark("  rhs + LU solve (all characters)");
        let mut ch = 0;
        while ch < NC as usize && tot > 0 {
            let chi = &g.chi[ch];
            let rhs = &sol[ch];
            for c in 0..nrhs as usize {
                for gi in 0..Gu {
                    for n in 0..Nsu {
                        self.phi[c * Nfu + g.img[gi * Nsu + n] as usize] += chi[gi] * R::c_as_cd(rhs[n + c * Nsu]);
                    }
                }
            }
            ch += 1;
        }

        // dampers: the solved psi_n becomes the jump mu = (-i eps / w) psi_n
        for j in 0..Nfu {
            if g.damper[j] >= 0 {
                let cj = Complex64::new(0.0, -g.dmp[g.damper[j] as usize].eps / self.w);
                for c in 0..nrhs as usize {
                    self.phi[c * Nfu + j] *= cj;
                }
            }
        }
        Ok(())
    }

    /// The direct method's extra columns, after the field points (whose pass completes the sums of
    /// E): −S u + E / g projected on each character as the incident term of the total formulation
    /// is, solved with the factors `solve` kept; their solution on the body points to the
    /// DirectFreq.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn solve_extra(&mut self) -> Result<()> {
        let (g, pl) = (self.g, self.pl);
        let d = self.ex.expect("the extra columns");
        let (Ns, NC, N2, inner, work) = (self.Ns, self.NC, self.N2, self.inner, self.work);
        let (Nsu, Nfu, Gu, nx) = (Ns as usize, self.Nf as usize, self.G as usize, d.ncols);
        for c in 0..nx {
            for j in 0..Nfu {
                self.ex_e[c * Nfu + j] += self.ex_raw[c * Nfu + j] / g.pg[j].area;
            }
        }
        let mut sol: Vec<Vec<Complex<R>>> = vec![Vec::new(); NC as usize];
        {
            let (Smat, Amat, piv, ux, ex_e, grav) = (&self.Smat, &self.Amat, &self.piv, &self.ux, &self.ex_e, pl.grav);
            let mut items: Vec<(usize, &mut Vec<Complex<R>>)> = sol.iter_mut().enumerate().collect();
            try_for_each_grouped(&mut items, inner.min(NC) as usize, |(chu, rhs)| {
                work.check_cancel()?;
                // item ch writes sol[ch] only
                let (chu, rhs) = (*chu, &mut **rhs);
                let chi = &g.chi[chu];
                let mut U = vec![Complex::<R>::default(); Nsu * nx];
                for cx in 0..nx {
                    for n in 0..Nsu {
                        let mut v = CZ;
                        for gi in 0..Gu {
                            v += chi[gi] * ux[cx * Nfu + g.img[gi * Nsu + n] as usize];
                        }
                        U[n + cx * Nsu] = R::c_from(-v / Gu as f64);
                    }
                }
                *rhs = vec![Complex::<R>::default(); Nsu * nx];
                R::gemm(Nsu, nx, Nsu, &Smat[chu * N2..(chu + 1) * N2], Nsu, &U, Nsu, rhs, Nsu);
                for cx in 0..nx {
                    for k in 0..Nsu {
                        let mut v = CZ;
                        for gi in 0..Gu {
                            v += chi[gi] * ex_e[cx * Nfu + g.img[gi * Nsu + k] as usize];
                        }
                        let r0 = R::c_as_cd(rhs[k + cx * Nsu]);
                        rhs[k + cx * Nsu] = R::c_from(r0 + v / (Gu as f64 * grav));
                    }
                }
                if g.wavemaker {
                    let s = R::from_f64(4.0 * PI);
                    for v in rhs.iter_mut() {
                        *v = *v / s;
                    }
                } else {
                    R::lu_substitute(Ns, &Amat[chu * N2..(chu + 1) * N2], &piv[chu], nx as i32, rhs, work)?;
                }
                Ok(())
            })?;
        }
        let mut phix = vec![CZ; nx * Nfu];
        for (ch, rhs) in sol.iter().enumerate() {
            let chi = &g.chi[ch];
            for cx in 0..nx {
                for gi in 0..Gu {
                    for n in 0..Nsu {
                        phix[cx * Nfu + g.img[gi * Nsu + n] as usize] += chi[gi] * R::c_as_cd(rhs[n + cx * Nsu]);
                    }
                }
            }
        }
        let Nb = pl.bp.len();
        let mut o = vec![CZ; nx * Nb];
        for c in 0..nx {
            for (q, &j) in pl.bp.iter().enumerate() {
                o[c * Nb + q] = phix[c * Nfu + j as usize];
            }
        }
        *d.out.lock().unwrap_or_else(|p| p.into_inner()) = o;
        self.prof.mark("  direct: the extra columns");
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn forces(&mut self) {
        let (g, pl, fc) = (self.g, self.pl, self.fc);
        let (Nf, nrad, nh, D, rho, w, limit) =
            (self.Nf as usize, self.nrad as usize, self.nh as usize, self.D as usize, self.rho, self.w, self.limit);
        let phi = &self.phi;
        let phiI = &self.phiI;
        for i in 0..D {
            let ci = &pl.coords[i];
            let bi = ci.body;
            for c in 0..nrad {
                let mut z = CZ;
                if ci.mode != 0 {
                    for j in 0..Nf {
                        if g.body[j] == bi && g.mirror[j] == 0 && g.body_surface(j as i32) && !g.free_damper(j as i32) {
                            z += self.ncoord(ci, j) * phi[c * Nf + j] * g.pg[j].area; // a sheet: the jump mu (theory §9.5)
                        }
                    }
                    z *= -rho;
                } else {
                    for j in 0..Nf {
                        z += self.pweight(ci, j) * phi[c * Nf + j];
                    }
                    z *= rho;
                }
                if limit {
                    let a = if fc.mode == FreqMode::Zero { &mut self.out.A0 } else { &mut self.out.Ainf };
                    a[i * nrad + c] = z.re as f32;
                    continue;
                }
                let o = i * nrad + c;
                self.out.A[o] = z.re as f32;
                self.out.B[o] = (-w * z.im) as f32;
            }
            for h in 0..nh {
                let mut x = CZ;
                let mut fk = CZ;
                if ci.mode == 0 {
                    for j in 0..Nf {
                        let wgt = self.pweight(ci, j);
                        if wgt == 0.0 {
                            continue;
                        }
                        let pd = if pl.total {
                            phi[(nrad + h) * Nf + j]
                        } else {
                            phi[(nrad + h) * Nf + j] + phiI[h * Nf + j]
                        };
                        x -= wgt * pd;
                        fk -= wgt * phiI[h * Nf + j];
                    }
                }
                if ci.mode != 0 {
                    for j in 0..Nf {
                        if g.body[j] == bi && g.mirror[j] == 0 && g.body_surface(j as i32) && !g.free_damper(j as i32) {
                            // on a sheet the incident wave has no jump: mu_D = mu_S, no Froude-Krylov part
                            let wet = g.kind[j] != SurfaceKind::Sheet; // a wetted element or a plate
                            let pd = if pl.total || !wet {
                                phi[(nrad + h) * Nf + j]
                            } else {
                                phi[(nrad + h) * Nf + j] + phiI[h * Nf + j]
                            };
                            let ni = self.ncoord(ci, j);
                            x += ni * pd * g.pg[j].area;
                            if wet {
                                fk += ni * phiI[h * Nf + j] * g.pg[j].area;
                            }
                        }
                    }
                }
                let o = h * D + i;
                self.out.X[o] = to_cf(self.iwr * x);
                self.out.FK[o] = to_cf(self.iwr * fk);
                self.out.SC[o] = to_cf(self.iwr * (x - fk));
            }
        }
        self.prof.mark("  forces (pressure)");
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn haskind(&mut self) {
        let (g, pl, fc) = (self.g, self.pl, self.fc);
        let (Nf, nrad, nh) = (self.Nf as usize, self.nrad as usize, self.nh as usize);
        let (phi, phiI, dphiI, LphiI) = (&self.phi, &self.phiI, &self.dphiI, &self.LphiI);
        // (no Haskind relation with a free damper, theory §9.5)
        if nh > 0 && nrad > 0 && !g.any_free_damper() {
            for c in 0..nrad {
                let bi = pl.rad[c].body;
                for h in 0..nh {
                    let mut v = CZ;
                    let col = &pl.rad[c];
                    for j in 0..Nf {
                        if g.mirror[j] != 0 || g.kind[j] == SurfaceKind::Interior {
                            continue;
                        }
                        if g.kind[j] == SurfaceKind::Sheet {
                            // a sheet: the n_i phi_I terms of its two faces cancel, dphi_I/dn changes
                            // sign, so the faces give -int mu_i dphi_I/dn with the jump mu_i
                            v -= phi[c * Nf + j] * dphiI[h * Nf + j] * g.pg[j].area;
                            continue;
                        }
                        if g.kind[j] == SurfaceKind::Press {
                            // on S_p dphi_I/dn becomes L phi_I and n_i the mode shape (theory §9.3),
                            // in the normals of this solver: -n_i phi_I + phi_i L phi_I
                            let ni = if col.mode == 0 && g.psurf[j] == col.ps {
                                press_shape(col.shape, &g.lc[j])
                            } else {
                                0.0
                            };
                            v += (-ni * phiI[h * Nf + j] + phi[c * Nf + j] * LphiI[h * Nf + j]) * g.pg[j].area;
                            continue;
                        }
                        if g.kind[j] == SurfaceKind::Lid {
                            // lid contribution (K - kappa) int phi_i phi_I dS
                            v += (fc.K - fc.kappa[g.lid[j] as usize])
                                * phi[c * Nf + j]
                                * phiI[h * Nf + j]
                                * g.pg[j].area;
                            continue;
                        }
                        let ni = if col.mode != 0 && g.body[j] == bi {
                            Complex64::new(self.ncoord(col, j), 0.0)
                        } else {
                            CZ
                        };
                        v += (ni * phiI[h * Nf + j] - phi[c * Nf + j] * dphiI[h * Nf + j]) * g.pg[j].area;
                    }
                    self.out.HK[c * nh + h] = to_cf(self.iwr * v);
                }
            }
        }

        self.prof.mark("  forces (Haskind)");
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn surface_elevations(&mut self) {
        let (g, pl, fc) = (self.g, self.pl, self.fc);
        let (Nf, nrad, nh, w) = (self.Nf as usize, self.nrad as usize, self.nh as usize, self.w);
        let (phi, phiI) = (&self.phi, &self.phiI);
        let out = &mut *self.out;
        // lid elevations eta = kappa phi / (i w) (theory §9.1): per unit displacement
        // (phi_R = i w phi_j) for radiation, total for diffraction
        if !self.limit && g.L > 0 {
            for j in 0..Nf {
                let q = g.lid_out[j];
                if q < 0 {
                    continue;
                }
                let q = q as usize;
                let kap = fc.kappa[g.lid[j] as usize];
                for c in 0..nrad {
                    out.lrad[q * nrad + c] = to_cf(kap * phi[c * Nf + j]);
                }
                for h in 0..nh {
                    let pd =
                        if pl.total { phi[(nrad + h) * Nf + j] } else { phi[(nrad + h) * Nf + j] + phiI[h * Nf + j] };
                    out.ldif[q * nh + h] = to_cf(kap * pd * Complex64::new(0.0, -1.0 / w)); // / (i w)
                }
            }
        }
        // pressure-surface elevations (theory §9.3): zeta_j = dphi_j/dz = K phi_j + n_j per unit
        // coordinate (radiation), zeta = K phi_D / (i w) (total diffraction)
        if !self.limit && g.Q > 0 {
            for j in 0..Nf {
                let q = g.press_out[j];
                if q < 0 {
                    continue;
                }
                let q = q as usize;
                for c in 0..nrad {
                    let col = &pl.rad[c];
                    let nj = if col.mode == 0 && g.psurf[j] == col.ps { press_shape(col.shape, &g.lc[j]) } else { 0.0 };
                    out.prad[q * nrad + c] = to_cf(fc.K * phi[c * Nf + j] + nj);
                }
                for h in 0..nh {
                    let pd =
                        if pl.total { phi[(nrad + h) * Nf + j] } else { phi[(nrad + h) * Nf + j] + phiI[h * Nf + j] };
                    out.pdif[q * nh + h] = to_cf(fc.K * pd * Complex64::new(0.0, -1.0 / w));
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn body_pressure(&mut self) {
        let pl = self.pl;
        let (Nf, nrad, nh, w) = (self.Nf as usize, self.nrad as usize, self.nh as usize, self.w);
        let (phi, phiI) = (&self.phi, &self.phiI);
        let out = &mut *self.out;
        // pressure p / (rho g) at the body panels (theory §4.1): w^2/g phi_j per unit displacement, -(i w/g) phi_D
        if pl.bpres && !self.limit {
            let Nb = pl.bp.len();
            for q in 0..Nb {
                let j = pl.bp[q] as usize;
                for c in 0..nrad {
                    out.bpr[q * nrad + c] = to_cf(w * w / pl.grav * phi[c * Nf + j]);
                }
                for h in 0..nh {
                    let pd =
                        if pl.total { phi[(nrad + h) * Nf + j] } else { phi[(nrad + h) * Nf + j] + phiI[h * Nf + j] };
                    out.bpd[q * nh + h] = to_cf(Complex64::new(0.0, -w / pl.grav) * pd);
                }
            }
        }
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn field_points(&mut self) -> Result<()> {
        let g = self.g;
        let (nrad, nh) = (self.nrad as usize, self.nh as usize);
        // field points
        let P = if self.limit { 0 } else { g.fp.len() };
        if P > 0 {
            let out = &mut *self.out;
            field_values::<R, Complex64>(
                g,
                self.pl,
                self.sf,
                self.fc,
                &g.fp,
                nrad as i32,
                nh as i32,
                &self.phi,
                &self.u,
                &self.phiI,
                &self.fpt,
                &mut out.frad[..],
                &mut out.finc[..],
                &mut out.fsca[..],
                &mut out.ftot[..],
                self.work,
                self.ex.map(|d| (d, &mut self.ex_raw[..])),
            )?;
        } else if self.ex.is_some() {
            return Err(crate::Error::internal("the direct method without its field points"));
        }
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn kochin(&mut self) -> Result<()> {
        let (g, pl, fc) = (self.g, self.pl, self.fc);
        let (Nf, nrad, nh) = (self.Nf as usize, self.nrad as usize, self.nh as usize);
        let (phi, phiI, u) = (&self.phi, &self.phiI, &self.u);
        let out = &mut *self.out;
        // Kochin functions of the solutions on the global angles (theory §11.1):
        // Green's identity with the far-field kernel over the wetted surface and the plates. A
        // lid panel would take the dipole column of a body panel here instead of the lid column
        // -(K - kappa) S, which is right only because route.rs refuses the mean drift (the only
        // use of the Kochin angles) with damping lids; the check keeps that precondition (O(Nf)
        // per frequency)
        if !self.limit && pl.NK > 0 {
            if g.kind[..Nf].contains(&SurfaceKind::Lid) {
                return Err(crate::Error::internal(
                    "Kochin functions with lid panels: route.rs refuses the mean drift with lids",
                ));
            }
            let NK = pl.NK as usize;
            let mut Sx = vec![CZ; Nf];
            let mut Dx = vec![CZ; Nf];
            let mut St = vec![CZ; Nf];
            let mut Dt = vec![CZ; Nf];
            for t in 0..NK {
                let th = 2.0 * PI * t as f64 / NK as f64;
                let tf = th - g.yaw * PI / 180.0;
                let c = tf.cos();
                let s = tf.sin();
                for j in 0..Nf {
                    Sx[j] = CZ;
                    Dx[j] = CZ;
                    St[j] = CZ;
                    Dt[j] = CZ;
                    if g.interior(j as i32) {
                        continue;
                    }
                    let p = &g.pg[j];
                    let f = far_kernel(fc, &p.c, &p.n, c, s);
                    Sx[j] = p.area * f.g;
                    Dx[j] = p.area * f.gn;
                    St[j] = p.area * f.gt;
                    Dt[j] = p.area * f.gnt;
                    if g.kind[j] == SurfaceKind::Press {
                        // the pressure column of kind_column (the interior panels are skipped
                        // above, lids absent)
                        Dx[j] = press_column(g, fc, j as i32, Sx[j], Dx[j]);
                        Dt[j] = press_column(g, fc, j as i32, St[j], Dt[j]);
                    }
                    if g.kind[j] == SurfaceKind::Plate {
                        // the plate column of kind_column: dZ/dn = -Z_z = -K Z on z = 0 (the
                        // normal -z into the water, K = k tanh kh), -K times the source integral
                        // (theory §11.1; no edge term: the far field needs no value at the edge)
                        Dx[j] = plate_column(fc, Sx[j]);
                        Dt[j] = plate_column(fc, St[j]);
                    }
                    if g.kind[j] == SurfaceKind::Sheet {
                        Sx[j] = CZ;
                        St[j] = CZ;
                    }
                }
                for c2 in 0..nrad {
                    let mut v = CZ;
                    let mut vt = CZ;
                    for j in 0..Nf {
                        v += phi[c2 * Nf + j] * Dx[j] - Sx[j] * u[c2 * Nf + j];
                        vt += phi[c2 * Nf + j] * Dt[j] - St[j] * u[c2 * Nf + j];
                    }
                    (v, vt) = kochin_global(fc, g.px, g.py, th, v, vt);
                    let o = t * nrad + c2;
                    out.kr[o] = to_cf(v);
                    out.krt[o] = to_cf(vt);
                }
                for h in 0..nh {
                    let mut v = CZ;
                    let mut vt = CZ;
                    for j in 0..Nf {
                        let mut ph = phi[(nrad + h) * Nf + j];
                        // the scattered formulation: the total potential with the dipole column
                        // alone (the incident wave's own Green identity over the closed surface
                        // vanishes); on a plate D phi_I equals -S dphi_S/dn, both -K S phi_I
                        if !pl.total
                            && (g.kind[j] == SurfaceKind::Wet
                                || g.kind[j] == SurfaceKind::Press
                                || g.kind[j] == SurfaceKind::Plate)
                        {
                            ph += phiI[h * Nf + j];
                        }
                        v += ph * Dx[j];
                        vt += ph * Dt[j];
                    }
                    (v, vt) = kochin_global(fc, g.px, g.py, th, v, vt);
                    let o = t * nh + h;
                    out.kd[o] = to_cf(v);
                    out.kdt[o] = to_cf(vt);
                }
            }
        }
        Ok(())
    }
}

/// The field outputs of one ordinary frequency at the points pts (frame coordinates) from the
/// surface solution: phi [nrad + nh][Nf], u [nrad][Nf] (the rows of the radiation
/// columns), phi_I [nh][Nf] on the panels; sf the static rows of pts, fpt the finite-depth
/// factors of the panels (a channel only). Writes frad [P][nrad] and finc, fsca, ftot [P][nh]
/// (elevation at z = 0, else p / (rho g)). The points on the threads of the current pool, each
/// point its own outputs (the results do not depend on their count); work: the cancel
/// check before every point.
/// T: the element type of the potentials (the solves use Complex64). ex: the direct method's
/// extra columns and their sums Σ_q ws_q S_qk [nx][Nf], added from the same rows (the points then in
/// batches of DIRECT_BATCH, the order of the sums fixed).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn field_values<R: Real, T: Copy + Into<Complex64> + Send + Sync>(
    g: &Geometry,
    pl: &Plan,
    sf: &Static<R>,
    fc: &FreqCtx,
    pts: &[Vec3],
    nrad: i32,
    nh: i32,
    phi: &[T],
    u: &[T],
    phiI: &[T],
    fpt: &[FinPoint],
    frad: &mut [Complex32],
    finc: &mut [Complex32],
    fsca: &mut [Complex32],
    ftot: &mut [Complex32],
    work: Work<'_>,
    ex: Option<(&crate::problem::DirectFreq, &mut [Complex64])>,
) -> Result<()> {
    let Nf = g.Nf as usize;
    let P = pts.len();
    let (nrad, nh) = (nrad as usize, nh as usize);
    let w = fc.w;
    let points_done = AtomicUsize::new(0);
    // the outputs of point qu (its own rows of frad, finc, fsca, ftot); full: also the source
    // integrals of the interior panels, and the row of S returned (the direct method's sums)
    let point = |qu: usize,
                 frad: &mut [Complex32],
                 finc: &mut [Complex32],
                 fsca: &mut [Complex32],
                 ftot: &mut [Complex32],
                 full: bool|
     -> Result<Option<Vec<Complex64>>> {
        work.check_cancel()?;
        work.sub(points_done.fetch_add(1, Ordering::Relaxed), P);
        let mut Sx = vec![CZ; Nf];
        let mut Dx = vec![CZ; Nf];
        let x = &pts[qu];
        let row = qu * Nf;
        let fx = if g.W > 0.0 { fin_point(&fc.fd, x[2]) } else { FinPoint::default() };
        // the rows of this point made now (rows too large to store) or the stored ones
        let own: Vec<StaticEntry<R>> = match &sf.lazy {
            Some(lz) => {
                let mut r = vec![StaticEntry::<R>::default(); Nf];
                static_row(g, x, lz.logint, lz.tol, lz.depth, false, &mut r, None);
                r
            }
            None => Vec::new(),
        };
        let view = StaticView::<R> { e: &own, inf: &[] };
        for j in 0..Nf {
            if g.interior(j as i32) && !full {
                continue;
            }
            if g.W > 0.0 {
                (Sx[j], Dx[j]) = channel_influence(g, sf, row, j as i32, x, &fx, &fpt[j], fc);
            } else if sf.lazy.is_some() {
                (Sx[j], Dx[j]) = influence(g, &view, 0, j as i32, x, fc);
            } else {
                (Sx[j], Dx[j]) = influence(g, sf, row, j as i32, x, fc); // Green's identity, S_b and lids
            }
        }
        let srow = if full { Some(Sx.clone()) } else { None };
        for j in 0..Nf {
            if full && g.interior(j as i32) {
                (Sx[j], Dx[j]) = (CZ, CZ); // the interior panels only in the sums of the direct method
            }
            if g.kind[j] == SurfaceKind::Sheet {
                Sx[j] = CZ; // a sheet carries dipoles only
            }
        }
        for c in 0..nrad {
            let mut v = CZ;
            for j in 0..Nf {
                v += phi[c * Nf + j].into() * Dx[j] - Sx[j] * u[c * Nf + j].into();
            }
            frad[c] = to_cf(w * w / pl.grav * v / (4.0 * PI));
        }
        for h in 0..nh {
            let mut v = CZ;
            // scattered formulation: the field of the total potential phi_S + phi_I
            // on the body and the plates, int (phi_I dG/dn - G dphi_I/dn) dS = 0; on a lid, whose
            // column holds the condition of the total potential, also phi_S + phi_I
            for j in 0..Nf {
                v += phi[(nrad + h) * Nf + j].into() * Dx[j];
                let k = g.kind[j];
                if !pl.total
                    && (k == SurfaceKind::Wet
                        || k == SurfaceKind::Press
                        || k == SurfaceKind::Lid
                        || k == SurfaceKind::Plate)
                {
                    v += phiI[h * Nf + j].into() * Dx[j];
                }
            }
            let phiS = v / (4.0 * PI);
            let (pI, _) = incident(g, pl, fc, pl.heads[h], x);
            let etaI = pI * Complex64::new(0.0, -w / pl.grav);
            let etaS = phiS * Complex64::new(0.0, -w / pl.grav);
            finc[h] = to_cf(etaI);
            fsca[h] = to_cf(etaS);
            ftot[h] = to_cf(etaI + etaS);
        }
        Ok(srow)
    };
    match ex {
        None => {
            // the points on the threads, each with its own influence rows and outputs
            let rows = par_rows(frad, nrad, P)
                .zip(par_rows(finc, nh, P))
                .zip(par_rows(fsca, nh, P).zip(par_rows(ftot, nh, P)));
            rows.enumerate().try_for_each(|(qu, ((frad, finc), (fsca, ftot)))| {
                point(qu, frad, finc, fsca, ftot, false).map(|_| ())
            })?;
        }
        Some((d, acc)) => {
            let nx = d.ncols;
            let mut q0 = 0;
            while q0 < P {
                let n = DIRECT_BATCH.min(P - q0);
                let rows = par_rows(&mut frad[q0 * nrad..], nrad, n)
                    .zip(par_rows(&mut finc[q0 * nh..], nh, n))
                    .zip(par_rows(&mut fsca[q0 * nh..], nh, n).zip(par_rows(&mut ftot[q0 * nh..], nh, n)));
                let got: Vec<Option<Vec<Complex64>>> = rows
                    .enumerate()
                    .map(|(i, ((frad, finc), (fsca, ftot)))| point(q0 + i, frad, finc, fsca, ftot, true))
                    .collect::<Result<Vec<_>>>()?;
                let batch: Vec<(Vec<Complex64>, Vec<Complex64>)> = got
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let ws = &d.ws[(q0 + i) * nx..(q0 + i + 1) * nx];
                        if ws.iter().all(|v| *v == CZ) { None } else { r.map(|r| (ws.to_vec(), r)) }
                    })
                    .collect();
                add_weighted_rows(acc, nx, Nf, &batch);
                q0 += n;
            }
        }
    }
    Ok(())
}

/// The field points of a batch of the direct method's sums (field_values, direct_gradients): the
/// order of the sums is that of the points, whatever the threads.
const DIRECT_BATCH: usize = 256;

/// e[c Nf + j] += Σ_p w_p[c] row_p[j] over the rows of a batch (the direct method's sums over the
/// field points), the columns c on the threads, the rows in their order.
fn add_weighted_rows(e: &mut [Complex64], nx: usize, nf: usize, rows: &[(Vec<Complex64>, Vec<Complex64>)]) {
    e[..nx * nf].par_chunks_mut(nf.max(1)).enumerate().for_each(|(c, ec)| {
        for (w, row) in rows {
            let wc = w[c];
            if wc == CZ {
                continue;
            }
            for (a, b) in ec.iter_mut().zip(row) {
                *a += wc * b;
            }
        }
    });
}

/// Solve one ordinary frequency or limit: assembly per symmetry character, LU, forces, motions,
/// fields, drift.
/// m problem, g geometry, pl solve plan; st static influence data of the body panels, sf of the
/// field points; fc kernel data of the frequency, scs the static rows of the control-surface
/// points; fi index of the ordinary frequency, or -1 for a limit (fc.mode), which solves the
/// radiation problems only and stores the added mass; work: its progress and cancel check; out
/// the outputs of the frequency (allocate_outputs of one frequency). The parallel loops run on
/// the current pool (the inner pool of the frequency).
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn solve_frequency<R: Real>(
    m: &Problem,
    g: &Geometry,
    pl: &Plan,
    st: &Static<R>,
    sf: &Static<R>,
    fc: &FreqCtx,
    fi: i32,
    work: Work<'_>,
    out: &mut Output,
    scs: Option<&Static<R>>,
) -> Result<()> {
    let mut f = Frequency::new(m, g, pl, st, sf, fc, work, out);
    f.boundary_values();
    if fi >= 0
        && let Some(d) = m.direct.as_ref().and_then(|d| d.freqs.get(fi as usize))
        && d.ncols > 0
    {
        f.ex = Some(d); // the direct method's extra columns, solved after the field points
    }
    f.matrices()?;
    if f.ex.is_some() {
        f.direct_gradients(m)?;
        if f.ex.is_some_and(|d| !d.bt.is_empty()) {
            f.direct_body_terms()?;
        }
    }
    f.sheet_rows()?;
    f.solve()?;
    // the output steps, each reported and preceded by a cancel check
    let step = |k: i32| -> Result<()> {
        work.check_cancel()?;
        work.report(Stage::Outputs, k, 6);
        Ok(())
    };
    step(0)?;
    f.forces();
    f.haskind();
    add_tank_terms(pl, fc.mode, fi, f.out); // internal tanks, before the motions
    step(1)?;
    f.surface_elevations();
    f.body_pressure();
    step(2)?;
    f.field_points()?;
    if f.ex.is_some() {
        f.solve_extra()?;
    }
    step(3)?;
    f.kochin()?;
    step(4)?;
    // motions (theory §3.1; D x D with the pressure coordinates, no mass on them)
    solve_motions(m, pl, f.w, f.nh, f.out, work)?;
    // the source formulation: field and body velocities, body pressure, the
    // pressure-integration drift (lo_velocity.rs)
    step(5)?;
    if f.srcv {
        let input = SourceInput {
            phi: &f.phi,
            u: &f.u,
            phiI: &f.phiI,
            dphiI: &f.dphiI,
            LphiI: &f.LphiI,
            Smat: &f.Smat,
            Vst: &f.Vst,
            fpt: &f.fpt,
            scs,
        };
        source_outputs::<R>(m, g, pl, sf, fc, fi, work, &input, &mut f.prof, f.out)?;
    }
    Ok(())
}
