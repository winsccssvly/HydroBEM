//! The memory estimate of a solve.
//!
//! The memory a solve will use, from the sizes of the problem without solving it: the
//! statics shared by all frequencies, the workspace of one frequency thread and the outputs (the
//! arrays of the solve's outputs, which the results take over, the quantities the results add and
//! the largest copy made while they are built, solve_driver::result_bytes).
//! The byte counts are sizes per element of the main arrays: those of their element types
//! (`size_of`), or the bytes of the numbers each element holds. Two totals: the resident
//! peak (the pages a solve writes: the arrays and the parts of the threads' packing buffers that the
//! factorization touches) and the commit charge (the arrays and the whole packing buffers, which
//! are reserved for the life of each thread that runs a matrix product). With the potential QTF
//! its two sub-problems are estimated the same way, and the arrays of the steps between them (the
//! fits, the outer integral, the pairs) are counted (second_order).

use crate::env_hooks::diag_double;
use crate::error::{Error, Result};
use crate::geometry::Vec3;
use crate::green_fin::FinPoint;
use crate::ho_geometry;
use crate::ho_geometry::projection_bytes;
use crate::ho_internal::{Pt, StaticRow};
use crate::interior::InteriorPanels;
use crate::linalg::{panel_resident_bytes, thread_workspace_bytes, update_resident_bytes};
use crate::lo_geometry::build_geometry;
use crate::low_order::SurfaceKind;
use crate::panel_integrals::PanelGeom;
use crate::problem::Problem;
use crate::route::{Route, select_route};
use crate::solve_common::base_plan;
use crate::solve_common::plan_coordinates;
use crate::solve_driver::{frequency_output_bytes, output_bytes, qtf_bytes, result_bytes};
use crate::solver::MemoryEstimate;

const CD: i64 = 16; // a complex float64

// The factorizations of one frequency: the size n of the matrices, the bytes s of an element and
// the factorizations that run at once (each on threads of the frequency's pool); and the body
// output points of the solve; the static rows of its field points that a later solve of the same
// points shares (Problem::field_statics: the low order's stored rows, none when they are lazy); the
// results the solve leaves (its outputs taken over and the quantities added, without the copy made
// while they are built), which a later solve of the potential QTF holds.
struct Factorizations {
    n: i64,
    s: i64,
    at_once: i64,
    body_points: i64,
    shared: i64,
    results: i64,
}

// The low order: the arrays of solver.rs, lo_static.rs, lo_frequency.rs and lo_velocity.rs.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn low_order(m: &Problem) -> Result<(MemoryEstimate, Factorizations)> {
    let mut ints: Vec<InteriorPanels> = Vec::new();
    let mut g = build_geometry(m, &mut ints)?;
    g.source = m.opt.pressure_drift || m.opt.field_velocity || m.opt.body_velocity || m.opt.control_surface_drift;
    let mut pl = base_plan(m);
    plan_coordinates(m, &g.ps, &mut pl);
    let R: i64 = if diag_double() { 8 } else { 4 }; // the precision of the matrices
    let C = 2 * R;
    let (Ns, Nf, NC, G) = (g.Ns as i64, g.Nf as i64, g.chi.len() as i64, g.G as i64);
    let (P, Pr) = (g.fp.len() as i64, g.cs_reps.len() as i64);
    let nrad = pl.rad.len() as i64;
    let nh = if pl.diffraction { pl.heads.len() as i64 } else { 0 };
    let nrhs = nrad + nh;
    let N2 = Ns * Ns;
    let inner = m.opt.inner_threads as i64;
    let want_inf = m.has_infinite() && pl.radiation;
    let channel = g.W > 0.0;
    let images: i64 = if channel { 3 } else { 1 }; // a channel: the entries at the points reflected in the walls
    let mut e = MemoryEstimate::default();
    // the statics (lo_static.rs): per row and panel the entries (4 R), at omega = inf two more,
    // the source-formulation gradients (2 x 3 R), the sheet rows (4 R, at omega = inf 2 R)
    let rows = |n: i64, inf: bool, source: bool| {
        let mut b = images * n * Nf * (4 + if inf { 2 } else { 0 }) * R;
        if source {
            b += images * n * Nf * 6 * R;
        }
        b
    };
    let mut sheets: i64 = 0;
    for k in 0..g.Ns as usize {
        sheets += (g.kind[g.sec[k] as usize] == SurfaceKind::Sheet) as i64;
    }
    // the velocity rows: the field points for their velocity, the control-surface points always
    // the field points' rows, or one row per thread when they are made at evaluation (lazy)
    let vfrom = (m.opt.field_velocity_from as i64).min(P);
    let vrows = if g.source && m.opt.field_velocity { P - vfrom } else { 0 };
    let lazy = crate::low_order::field_statics_lazy(P as usize, Nf as usize, vrows as usize, R as usize, channel);
    let field_rows = if lazy {
        m.opt.total_threads() as i64 * Nf * 10 * R
    } else {
        images * P * Nf * 4 * R + images * vrows * Nf * 6 * R
    };
    e.statics = rows(Ns, want_inf, g.source)
        + field_rows
        + rows(Pr, false, g.source && m.opt.control_surface_drift)
        + sheets * Nf * (if want_inf { 6 } else { 4 }) * R;
    // the field points in the frame (Geometry::fp)
    e.statics += P * size_of::<Vec3>() as i64;
    // the panels (with their float32 tier rules); tq counted at its length
    for p in &g.pg {
        e.statics += (size_of::<PanelGeom>() + p.tq.len() * size_of::<f32>()) as i64;
    }
    // one frequency (lo_frequency.rs): the character matrices, the gradients of the source
    // formulation, the incident wave and potentials, the characters solved at once (the right-hand
    // sides each; the LU in place of the matrix), the solutions of all characters
    let srcv = g.source && (((pl.pdrift || pl.cdrift) && nh > 0) || pl.fvel || pl.bvel);
    let mut f = 2 * NC * N2 * C;
    if srcv {
        f += Ns * Nf * 3 * C;
    }
    f += (if g.ps.is_empty() { 2 } else { 3 }) * nh * Nf * CD + 2 * nrhs * Nf * CD;
    if m.env.depth < f64::INFINITY {
        f += Nf * size_of::<FinPoint>() as i64;
    }
    f += NC * Ns * nrhs * C + NC.min(inner) * (2 * Ns * nrhs) * C;
    if srcv {
        // lo_velocity.rs: boundary values and sources, the source matrices and one copy, the
        // panel velocities and their products, the control-surface fields
        let nsr = nrad + nh;
        f += 2 * nsr * Nf * CD + (NC + 1) * N2 * C + Ns * nsr * C;
        if (pl.pdrift && nh > 0) || pl.bvel {
            f += nsr * Nf * 3 * CD + Nf * G * nsr * CD + 96 * (Nf + G * nsr) * CD;
        }
        if pl.cdrift && nh > 0 {
            f += 4 * Pr * G * nsr * CD + 4 * Nf * CD;
        }
    }
    e.per_frequency = f;
    let mut Nb: i32 = 0;
    if pl.bpres || pl.bvel {
        for j in 0..g.Nf as usize {
            Nb +=
                ((g.kind[j] == SurfaceKind::Wet || g.kind[j] == SurfaceKind::Press || g.kind[j] == SurfaceKind::Plate)
                    && g.mirror[j] == 0
                    && g.body[j] >= 0) as i32;
        }
    }
    e.outputs = output_bytes(m, &pl, P as i32, g.L, g.Q, Nb);
    // the outputs of the frequency on the thread, written into those of all when it is done
    e.per_frequency += frequency_output_bytes(m, &pl, P as i32, g.L, g.Q, Nb);
    if pl.pqtf.any() {
        let mut nw: i64 = 0;
        for j in 0..g.Nf as usize {
            nw += (g.kind[j] == SurfaceKind::Wet && g.mirror[j] == 0 && g.body[j] >= 0) as i64;
        }
        e.outputs += qtf_bytes(m, &pl, nw);
    }
    // the specified body points, interpolated from the panels' values (solver::put_specified_body_points)
    let specified = (pl.bpres || pl.bvel) && m.bodies.first().is_some_and(|b| !b.body_points.is_empty());
    let Np = if specified { m.bodies.iter().map(|b| b.body_points.len() as i64).sum() } else { 0 };
    let (added, copy) = result_bytes(m, &pl, P, Np);
    let results = e.outputs + added;
    e.outputs = results + copy;
    let shared = if lazy { 0 } else { field_rows };
    Ok((e, Factorizations { n: Ns, s: C, at_once: NC.min(inner), body_points: Nb as i64, shared, results }))
}

// The higher order: the arrays of ho_solver.rs and ho_frequency.rs.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn higher_order(m: &Problem) -> Result<(MemoryEstimate, Factorizations)> {
    let h = ho_geometry::build(m)?;
    let mut pl = base_plan(m);
    plan_coordinates(m, &h.ps, &mut pl);
    let (N, Ncol, NC) = (h.Ndof as i64, h.Ncol as i64, h.chi.len() as i64);
    let (NQ, No) = (h.inner.len() as i64, h.outer.len() as i64);
    let (Ninst, NM, P) = (h.inst.len() as i64, h.NM as i64, h.fp.len() as i64);
    let nrad = pl.rad.len() as i64;
    let nh = if pl.diffraction { pl.heads.len() as i64 } else { 0 };
    let nrhs = nrad + nh;
    let inner = m.opt.inner_threads as i64;
    let want_inf = m.has_infinite() && pl.radiation;
    // the geometry: the outer and inner points, the compact arrays of the wave rows (inner and
    // far rule, about as many far points as inner ones)
    let mut e = MemoryEstimate {
        statics: (NQ + No) * size_of::<Pt>() as i64 + 2 * NQ * (6 + NM + 36) * 8,
        ..MemoryEstimate::default()
    };
    // the mid and far points of every (instance, panel) block (build_blocks)
    let blocks = h.inner0.len() as i64 - 1;
    e.statics += blocks
        * (crate::ho_internal::MID_ORDER as i64 * crate::ho_internal::MID_ORDER as i64
            + crate::ho_internal::FAR_ORDER as i64 * crate::ho_internal::FAR_ORDER as i64)
        * size_of::<Pt>() as i64;
    // the static rows of the outer and field points (ho_static.rs)
    // (dense rs, rd, ms and at omega = inf rsi, rdi, msi; the log corrections sparse: their
    // dense form only while a row is built, one per thread)
    // (none for the outer points of sheets, which take collocation rows)
    let mut sheets: i64 = 0;
    for o in &h.outer {
        sheets += (h.kind_of(o.inst) == SurfaceKind::Sheet) as i64;
    }
    e.statics +=
        (No - sheets) * ((if want_inf { 4 } else { 2 }) * Ncol + (if want_inf { 2 } else { 1 }) * Ninst * NM) * 8;
    e.statics += P * (2 * Ncol + Ninst * NM) * 8;
    // the rows themselves (one StaticRow per outer and field point: its vectors' headers) and the
    // field points with their velocity stencils in the frame (HoGeom::fp)
    e.statics += (No + P) * size_of::<StaticRow>() as i64 + P * size_of::<Vec3>() as i64;
    e.statics += m.opt.total_threads() as i64 * 2 * (2 * Ncol + Ninst * NM) * 8;
    // the factored projections of the scattered diffraction, one per patch
    if nh > 0 && !pl.total {
        e.statics += projection_bytes(&h);
    }
    e.statics += sheets * (if want_inf { 2 } else { 1 }) * (2 * Ncol + Ninst * NM) * CD;
    // one frequency: the Galerkin system of every character, the characters solved at once
    // (their column-major copies), the incident wave, the scattered boundary values, the
    // coefficients and inner values, the field values, the row data of the body rows
    let mut f = NC * N * N * CD + NC * N * nrhs.max(1) * CD + NC.min(inner) * N * N * CD;
    f += 2 * nh * NQ * CD;
    if !pl.total {
        f += nh * (NC * N + Ncol) * CD;
    }
    f += nrhs * (Ncol + NQ) * CD + P * nrhs * CD;
    f += 4 * inner * (NC * N + nrad + NC * nh) * CD + inner * (2 * Ncol + Ninst * NM) * CD;
    e.per_frequency = f;
    e.outputs = output_bytes(m, &pl, h.Pu, h.L, h.Q, h.bpts.len() as i32);
    // the outputs of the frequency on the thread, written into those of all when it is done
    e.per_frequency += frequency_output_bytes(m, &pl, h.Pu, h.L, h.Q, h.bpts.len() as i32);
    if pl.pqtf.any() {
        let mut nw: i64 = 0;
        for q in &h.inner {
            nw += (h.kind_of(q.inst) == SurfaceKind::Wet && !h.inst[q.inst as usize].image) as i64;
        }
        e.outputs += qtf_bytes(m, &pl, nw);
    }
    // (the body points of the higher order are its own points, taken over without interpolation)
    let (added, copy) = result_bytes(m, &pl, h.Pu as i64, 0);
    let results = e.outputs + added;
    e.outputs = results + copy;
    let lu =
        Factorizations { n: N, s: CD, at_once: NC.min(inner), body_points: h.bpts.len() as i64, shared: 0, results };
    Ok((e, lu))
}

/// The memory estimate of a model (validated first): its own solve and, with the potential QTF,
/// the larger peak of its second order (second_order); the tank problems are not included. A hook
/// of the environment that is refused (env_hooks::environment_hooks) refuses the estimate as the
/// solve. The problem is `&mut` for the cache of its rules alone (`Problem::validate`).
pub fn memory_estimate(m: &mut Problem) -> Result<MemoryEstimate> {
    crate::env_hooks::environment_hooks()?;
    m.validate()?;
    if select_route(m)? == Route::Hydrostatics {
        return Ok(MemoryEstimate::default());
    }
    let (mut e, lu) = solve_estimate(m)?;
    if m.opt.potential_qtf.any() {
        second_order(m, lu.results, &mut e)?;
    }
    Ok(e)
}

/// The potential QTF's second order after the model's solve (potential_qtf::run), every step with
/// the model's results and the zones and field points of the free surface (Footprint::points) held:
/// the first-order sub-problem (a solve); then with its results and the
/// static rows of the field points it shares with the second sub-problem held, for the indirect
/// method the radiation at the pair frequencies (a solve), the fits (with the radiation's results
/// and the fields read from both), the outer integral (a batch of radii), the pairs (their
/// coefficients and the direct method's columns, and on each of the min(frequency_threads, pair
/// frequencies) workers a pair frequency at once with its work items, potential_qtf::Footprint),
/// and for the direct method the solve of the second sub-problem again with its extra columns (the
/// weights on the field points and the body condition of every pair, its solution) while the fields
/// of the fits stay held. The peak and the commit charge are the largest of the model's own and
/// these; the statics, the workspace and the outputs stay those of the model's own solve. model: the
/// results of the model's solve (Factorizations.results).
fn second_order(m: &Problem, model: i64, e: &mut MemoryEstimate) -> Result<()> {
    let sub = crate::potential_qtf::sub_problems(m)?;
    // the zones and the field points, held through every step with the model's results
    let model = model + sub.footprint.points;
    let (mut peak, mut commit) = (e.peak, e.commit);
    let mut at = |p: i64, c: i64| {
        peak = peak.max(p);
        commit = commit.max(c);
    };
    let (ea, la) = solve_estimate(&sub.problems[0])?;
    at(model + ea.peak, model + ea.commit);
    if let Some(sb) = sub.problems.get(1) {
        let (eb, lb) = solve_estimate(sb)?;
        let held = model + la.results; // and the first-order results
        let rb = if sub.indirect { lb.results } else { 0 };
        if sub.indirect {
            at(held + eb.peak, held + eb.commit);
        }
        let (f, i) = (m.opt.frequency_threads as i64, m.opt.inner_threads as i64);
        let st = sub.footprint.steps(la.body_points, rb, f, i);
        let held_rows = held + la.shared; // the field rows shared by both sub-problems
        for x in [st.fits, st.outer, st.pairs] {
            at(held_rows + x, held_rows + x);
        }
        if sub.direct_columns > 0 {
            let (n, p, v, nb) =
                (sub.direct_columns as i64, sub.field_points as i64, sub.velocity_points as i64, lb.body_points);
            let columns = CD * n * (p + 2 * v + 2 * nb + if sub.moving { 3 * nb } else { 0 });
            // the second sub-problem's statics hold the shared field rows again
            let d = held + st.through_direct + columns;
            at(d + eb.peak, d + eb.commit);
        }
    }
    e.peak = peak;
    e.commit = commit;
    Ok(())
}

/// The estimate of one solve of a valid model (not the hydrostatics route), with its factorizations.
fn solve_estimate(m: &Problem) -> Result<(MemoryEstimate, Factorizations)> {
    let (mut e, lu) = if m.opt.higher_order { higher_order(m)? } else { low_order(m)? };
    // the packing buffers of the matrix products on the threads of a frequency: committed whole on
    // its inner threads, which run the blocks of the LU, and on the frequency's own thread; resident
    // the parts the blocks of the factorizations write (the trailing updates on every inner thread,
    // the panels on the thread of each factorization)
    let inner = m.opt.inner_threads as i64;
    let committed = (inner + 1) * thread_workspace_bytes();
    let resident = inner * update_resident_bytes(lu.n, lu.s) + lu.at_once * panel_resident_bytes(lu.n, lu.s);
    e.per_frequency += resident;
    let threads = m.opt.frequency_threads as i64;
    e.peak = e.statics + threads * e.per_frequency + e.outputs;
    e.commit = e.peak + threads * (committed - resident).max(0);
    Ok((e, lu))
}

/// The check of a solve's memory before it starts (options.memory_check): the resident peak of
/// the estimate against the physical memory available now. A solve that needs more is refused with
/// ErrorKind::Value (the problem is too large for this machine, as a size beyond int32 is a value
/// error), the message giving both and the remedies; it would otherwise abort the process on the
/// first allocation that fails. Unknown available memory (no reading on this system) skips the
/// check. The environment variable HB_AVAILABLE_MEMORY (bytes, a test hook) replaces the reading,
/// read at every check so that a test can set it in the process; a value that is not a number is
/// refused (ErrorKind::Value), not passed over. The estimate only: the checks of the model before a
/// solve, the zones of the potential QTF's free surface among them, are `Model::pre_solve`'s.
pub(crate) fn check_memory(m: &mut Problem) -> Result<()> {
    let available = crate::env_hooks::env_number::<u64>("HB_AVAILABLE_MEMORY")?.or_else(available_memory);
    let Some(available) = available else {
        return Ok(());
    };
    let need = memory_estimate(m)?.peak.max(0) as u64;
    memory_refusal(need, available).map_or(Ok(()), |text| Err(Error::value(text)))
}

/// The refusal of a solve whose resident peak `need` exceeds the `available` bytes, None when it fits.
pub(crate) fn memory_refusal(need: u64, available: u64) -> Option<String> {
    (need > available).then(|| {
        format!(
            "the solve needs about {} of memory (the resident peak of memory_estimate) but {} are available; \
             use fewer options.frequency_threads, fewer panels or a coarser free surface \
             (second_order.free_surface_panel_size), or set options.memory_check = false to solve anyway",
            bytes_text(need),
            bytes_text(available)
        )
    })
}

// a byte count in GiB (at least 1 GiB) or MiB, two decimals
fn bytes_text(b: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    let x = b as f64;
    if x >= 1024.0 * MIB { format!("{:.2} GiB", x / (1024.0 * MIB)) } else { format!("{:.2} MiB", x / MIB) }
}

/// The physical memory available now (bytes): Windows GlobalMemoryStatusEx (ullAvailPhys); None
/// when the call fails.
#[cfg(windows)]
pub(crate) fn available_memory() -> Option<u64> {
    #[repr(C)]
    #[derive(Default)]
    // the MEMORYSTATUSEX of the Windows API
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, ..Default::default() };
    // SAFETY: GlobalMemoryStatusEx writes one MEMORYSTATUSEX (two DWORDs and seven DWORDLONGs, the
    // layout of the struct above, repr(C)) through a pointer to a live, writable value whose length
    // field holds its size, as the call requires; a zero return (failure) leaves it unread.
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) } != 0;
    ok.then_some(s.avail_phys)
}

/// The physical memory available now (bytes): MemAvailable of /proc/meminfo (Linux; this target is
/// not tested); None where it cannot be read.
#[cfg(not(windows))]
pub(crate) fn available_memory() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_solve_beyond_the_available_memory_is_refused() {
        assert!(memory_refusal(100, 100).is_none());
        let text = memory_refusal(3 << 30, 512 << 20).expect("refused");
        assert!(text.contains("3.00 GiB") && text.contains("512.00 MiB"), "{text}");
        assert!(text.contains("frequency_threads") && text.contains("options.memory_check = false"), "{text}");
        assert!(available_memory().is_some_and(|b| b > 0));
    }
}
