//! Internal tanks (theory §10.2): each
//! tank is a fluid domain of its own, bounded by its elements of the body mesh and its free
//! surface z = Z_T, with no influence between the domains. Its radiation problem is solved as a
//! problem of its own (the tank elements, the free surface moved to z = 0, the tank density) and
//! its added mass and damping join those of the body. Field points inside a tank are field
//! points of its problem.

use crate::codes::QuantityId;
use crate::constants::PI;
use crate::error::{Error, Result};
use crate::geometry::{Vec3, length_tol, norm, v3};
use crate::green_fin::FreqMode;
use crate::low_order::{Output, Plan, TankDrift, TankField};
use crate::mesh::{Mesh, Range, spline_layout};
use crate::mesh_desc::PatchKind;
use crate::model_codes::{Domain, Formulation, QtfHeadings};
use crate::pressure_drift::PressureDrift;
use crate::problem::{Body, GeneralizedMode, Problem};
use crate::progress::SolveContext;
use crate::quadratic_qtf::QtfRecorder;
use crate::results::{Element, Results, elements_of};
use crate::solve_common::base_plan;
use crate::solve_common::plan_coordinates;
use crate::solver::solve;
use crate::text::element_text;
use num_complex::{Complex32, Complex64};

// The stored elements of mesh m listed in recs, as a mesh of their own (low order: the panels;
// higher order: the patch data of every geometry, a trimmed mesh's elements with its source).
fn subset_mesh(m: &Mesh, recs: &[i32]) -> Mesh {
    let mut s = Mesh {
        patch: m.patch,
        geometry: m.geometry,
        reference_length: m.reference_length,
        gravity: m.gravity,
        isx: m.isx,
        isy: m.isy,
        source: m.source.clone(),
        trim_r: m.trim_r,
        trim_z: m.trim_z,
        ..Mesh::default()
    };
    // the offsets of the spline data of every patch
    let (ko, co) = (&m.layout.knot_offset, &m.layout.coef_offset);
    for &r in recs {
        let r = r as usize;
        s.shapes.push(m.shapes[r].clone());
        if !m.patch {
            s.panels.push(m.panels[r]);
            continue;
        }
        s.discretization.push(m.discretization[r]);
        if m.trimmed() {
            s.trim.push(m.trim[r]);
            s.trim_root.push(m.trim_root[r].clone());
        } else if m.geometry == PatchKind::Flat {
            s.vertices.extend_from_slice(&m.vertices[12 * r..12 * (r + 1)]);
        } else if m.geometry == PatchKind::Analytic {
            s.analytic.push(m.analytic[r]);
        } else {
            s.orders.extend_from_slice(&m.orders[2 * r..2 * r + 2]);
            s.ncoef.extend_from_slice(&m.ncoef[2 * r..2 * r + 2]);
            s.knots.extend_from_slice(&m.knots[ko[r] as usize..ko[r + 1] as usize]);
            s.coefficients.extend_from_slice(&m.coefficients[3 * co[r] as usize..3 * co[r + 1] as usize]);
            if !m.weights.is_empty() {
                s.weights.extend_from_slice(&m.weights[co[r] as usize..co[r + 1] as usize]);
            }
        }
    }
    s.count = recs.len() as i32;
    if s.patch && s.geometry == PatchKind::Spline && !s.trimmed() {
        s.layout = spline_layout(&s);
    }
    s
}

// Element ranges of the original mesh mapped onto the elements of the subset (one-based).
fn remap(sel: &[Range], recs: &[i32]) -> Vec<Range> {
    let mut out: Vec<Range> = Vec::new();
    for (k, &rec) in recs.iter().enumerate() {
        let r = rec + 1;
        let mut in_ = false;
        for q in sel {
            in_ = in_ || (r >= q.first && r <= q.last);
        }
        if !in_ {
            continue;
        }
        let i = k as i32 + 1;
        match out.last_mut() {
            Some(b) if b.last == i - 1 => b.last = i,
            _ => out.push(Range { first: i, last: i }),
        }
    }
    out
}

// The values of quantity q of a results set as floats, doubles or ints (empty when absent).
fn read<T: Element>(r: &Results, q: QuantityId) -> Vec<T> {
    r.quantities.get(&q).map(|v| elements_of(v.data())).unwrap_or_default()
}
fn read_f32(r: &Results, q: QuantityId) -> Vec<f32> {
    read(r, q)
}
fn read_f64(r: &Results, q: QuantityId) -> Vec<f64> {
    read(r, q)
}
fn read_i32(r: &Results, q: QuantityId) -> Vec<i32> {
    read(r, q)
}

/// The added mass and damping of the internal tanks of every body: their sub-problems
/// solved in the context of the model's solve (its cancel request; no progress reports).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn plan_tanks(m: &Problem, pl: &mut Plan, ctx: &SolveContext) -> Result<()> {
    let F = pl.omegas.len();
    let D = pl.D as usize;
    let K = pl.rad.len();
    let any = m.bodies.iter().any(|b| !b.tanks.is_empty());
    if !any || K == 0 {
        return Ok(());
    }
    pl.tA = vec![0.0; F * D * K];
    pl.tB = vec![0.0; pl.tA.len()];
    pl.tA0 = vec![0.0; D * K];
    pl.tAinf = vec![0.0; pl.tA0.len()];
    let mut tank_no = 0; // the tanks of the bodies in order, one-based (the field-point tank numbers)
    for bi in 0..m.bodies.len() {
        let b = &m.bodies[bi];
        let d = &m.derived()[bi];
        let mesh = &m.meshes[b.mesh as usize];
        for ti in 0..b.tanks.len() {
            tank_no += 1;
            if b.tanks[ti].density == 0.0 {
                continue;
            }
            let recs: Vec<i32> = (0..mesh.count).filter(|&r| d.tank[r as usize] == ti as i32).collect();
            let (mut sub, pts) = tank_problem(m, bi, ti, &recs, tank_no);
            sub.validate()?;
            // the sub-problem stops on a cancel of the model's solve
            let r = solve(&mut sub, &SolveContext::new(ctx.cancel.clone(), None))?;
            // map the tank coordinates and columns onto the body's
            let mut sp = base_plan(&sub);
            plan_coordinates(&sub, &[], &mut sp);
            let col = tank_columns(pl, &sp, d.coordinate_offset)?;
            if !pts.is_empty() {
                map_tank_fields(pl, &r, &pts, &col);
            }
            if m.opt.pressure_drift && !m.opt.higher_order {
                // the pressure drift (low order): the tank problem gave the wall potentials
                let td = tank_drift(m, pl, bi, ti, &recs, &r, &col);
                pl.tdrift.push(td);
            }
            add_tank_coefficients(m, pl, bi, ti, &r, &sp, &col);
        }
    }
    Ok(())
}

/// The tank problems of the tanks with plates of their free surface (theory §10.2), with their
/// elements and their one-based number over the bodies: (body, tank, problem).
fn plated_tank_problems(m: &Problem) -> Vec<(usize, usize, Problem)> {
    let mut out = Vec::new();
    let mut tank_no = 0;
    for (bi, b) in m.bodies.iter().enumerate() {
        let d = &m.derived()[bi];
        let mesh = &m.meshes[b.mesh as usize];
        for (ti, t) in b.tanks.iter().enumerate() {
            tank_no += 1;
            if t.plates.is_empty() || t.density == 0.0 {
                continue;
            }
            let recs: Vec<i32> = (0..mesh.count).filter(|&r| d.tank[r as usize] == ti as i32).collect();
            out.push((bi, ti, tank_problem(m, bi, ti, &recs, tank_no).0));
        }
    }
    out
}

/// The model rules of the tank problems of the tanks with plates (the rules of free-surface plates
/// in the tank's frame: no overlap, the sheets meeting a plate on its edges, ...), so that the
/// model's check reports them before a solve.
pub fn check_tank_plates(m: &Problem) -> Result<()> {
    for (_, _, mut sub) in plated_tank_problems(m) {
        sub.check_rules()?;
    }
    Ok(())
}

/// The refusals of the plates of a tank's free surface (theory §10.2): the drift of the tank fluid
/// (low order, the pressure drift) has no term for a plate on the tank's waterline, and every
/// refusal of the tank problem's route (a plate there is a plate of the sea problem).
pub fn tank_plates_unsupported(m: &Problem) -> Result<()> {
    for (bi, ti, mut sub) in plated_tank_problems(m) {
        let b = &m.bodies[bi];
        if m.opt.pressure_drift || m.opt.pressure_qtf.any() {
            return Err(Error::unsupported(format!(
                "no solver route for this problem in this build: {}, {}: the mean drift by pressure \
                 integration (and its QTFs) with plates of a tank's free surface is not available",
                crate::problem::body_label(b),
                element_text("tank", crate::problem::model_place(b.tanks[ti].place, ti), &b.tanks[ti].name)
            )));
        }
        sub.route()?;
    }
    Ok(())
}

// The radiation problem of tank ti of body bi (its elements recs, one-based number tank_no over
// the bodies): its elements with their roles, the free surface at z = 0, the body's modes and
// generalized modes, radiation only, the tank density; the field points in the tank (pts, their
// indices among the model's) and, for the low-order pressure drift, the potential and velocity at
// the tank walls (the body pressure and velocity outputs).
fn tank_problem(m: &Problem, bi: usize, ti: usize, recs: &[i32], tank_no: i32) -> (Problem, Vec<usize>) {
    let b = &m.bodies[bi];
    let d = &m.derived()[bi];
    let t = &b.tanks[ti];
    let mut sub = Problem::default();
    sub.meshes.push(subset_mesh(&m.meshes[b.mesh as usize], recs));
    let mut tb = Body {
        name: format!("{} {}", b.name, element_text("tank", crate::problem::model_place(t.place, ti), &t.name)),
        mesh: 0,
        position: v3(b.position[0], b.position[1], -d.tank_z[ti]),
        yaw: b.yaw,
        rotation_center: b.rotation_center,
        modes: b.modes.clone(),
        ..Body::default()
    };
    for g0 in &b.generalized {
        let mut g: GeneralizedMode = g0.clone();
        for p in g.pieces.iter_mut() {
            if !p.panels.is_empty() {
                p.panels = remap(&p.panels, recs);
                if p.panels.is_empty() {
                    p.terms.clear(); // the piece is not on this tank
                }
            }
        }
        tb.generalized.push(g);
    }
    tb.dipole_panels = remap(&b.dipole_panels, recs);
    for dm in &b.dampers {
        let mut q = dm.clone();
        q.panels = remap(&dm.panels, recs);
        if !q.panels.is_empty() {
            tb.dampers.push(q);
        }
    }
    // the plates of the tank's free surface: plates of the tank body (z = Z_T at z = 0), its
    // elements remapped or a mesh of their own (body coordinates, placed with the tank body)
    for p0 in &t.plates {
        let mut p = p0.clone();
        if let Some(pi) = p.mesh {
            sub.meshes.push(m.meshes[pi].clone());
            p.mesh = Some(sub.meshes.len() - 1);
        } else {
            p.panels = remap(&p.panels, recs);
        }
        tb.plates.push(p);
    }
    sub.bodies.push(tb);
    sub.env = m.env;
    sub.env.depth = f64::INFINITY; // a closed domain: any Green function with the free-surface condition
    sub.interior_domain = true;
    sub.env.density = m.env.density * t.density;
    sub.opt = m.opt;
    sub.opt.diffraction = false;
    sub.opt.rao = false;
    sub.opt.drift_angles = 0;
    sub.opt.pressure_drift = false;
    sub.opt.pressure_qtf = QtfHeadings::None;
    sub.opt.control_surface_qtf = false;
    sub.opt.potential_qtf = QtfHeadings::None;
    sub.opt.field_pressure = false;
    sub.opt.field_velocity = false;
    sub.opt.body_pressure = false;
    sub.opt.body_velocity = false;
    // the tank fluid sees no walls of the sea: a channel's items leave with its domain
    sub.opt.domain = Domain::OpenSea;
    sub.opt.channel_width = f64::NAN;
    sub.opt.channel_reflect = 1.0;
    sub.opt.formulation = Formulation::Scattered;
    sub.omegas = m.omegas.clone();
    // its field points: body-local heights less the tank free surface
    let mut pts: Vec<usize> = Vec::new();
    for q in 0..m.field_tanks.len() {
        if m.field_tanks[q] == tank_no {
            pts.push(q);
            let x = &m.field_points[q];
            sub.field_points.push(v3(x[0], x[1], x[2] - b.position[2] - d.tank_z[ti]));
        }
    }
    if !pts.is_empty() {
        sub.has_field_points = true;
        sub.tank_domain = true;
        sub.opt.field_pressure = m.opt.field_pressure;
        sub.opt.field_velocity = m.opt.field_velocity;
    }
    // the pressure drift (low order): the potential and velocity at the tank walls
    if m.opt.pressure_drift && !m.opt.higher_order {
        sub.opt.body_pressure = true;
        sub.opt.body_velocity = true;
    }
    (sub, pts)
}

// The body column of every radiation column of the tank problem (its coordinates offset by the
// body's first coordinate off).
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn tank_columns(pl: &Plan, sp: &Plan, off: i32) -> Result<Vec<usize>> {
    let mut col: Vec<i32> = vec![-1; sp.rad.len()];
    for c in 0..sp.rad.len() {
        for k in 0..pl.rad.len() {
            if pl.rad[k].coord == off + sp.rad[c].coord {
                col[c] = k as i32;
            }
        }
    }
    // before the columns are used
    if col.iter().any(|&c| c < 0) {
        return Err(Error::internal("tank radiation column without a body column"));
    }
    Ok(col.iter().map(|&c| c as usize).collect())
}

// The radiation pressure and velocity of the field points pts of a tank problem (results r),
// [F][Pt][Ks] and [F][Pt][Ks][3], onto the body's columns.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn map_tank_fields(pl: &mut Plan, r: &Results, pts: &[usize], col: &[usize]) {
    let (F, K, Ks) = (pl.omegas.len(), pl.rad.len(), col.len());
    let fr = read_f32(r, QuantityId::FieldPointsPressureRadiation);
    let fv = read_f32(r, QuantityId::FieldPointsVelocityRadiation);
    let Pt = pts.len();
    for q in 0..Pt {
        let mut tf = TankField {
            point: pts[q] as i32,
            rad: vec![Complex32::new(0.0, 0.0); if fr.is_empty() { 0 } else { F * K }],
            vel: vec![Complex32::new(0.0, 0.0); if fv.is_empty() { 0 } else { F * K * 3 }],
        };
        for f in 0..F {
            for c in 0..Ks {
                let os = (f * Pt + q) * Ks + c;
                if !fr.is_empty() {
                    tf.rad[f * K + col[c]] = Complex32::new(fr[2 * os], fr[2 * os + 1]);
                }
                if !fv.is_empty() {
                    for e in 0..3 {
                        tf.vel[(f * K + col[c]) * 3 + e] =
                            Complex32::new(fv[2 * (os * 3 + e)], fv[2 * (os * 3 + e) + 1]);
                    }
                }
            }
        }
        pl.tfield.push(tf);
    }
}

// The pressure-drift data of tank ti of body bi from its tank problem (results r): the wall points
// in body axes, their potentials and velocities per unit motion on the body's columns, and the
// tank waterline.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn tank_drift(m: &Problem, pl: &Plan, bi: usize, ti: usize, recs: &[i32], r: &Results, col: &[usize]) -> TankDrift {
    let b = &m.bodies[bi];
    let zt = m.derived()[bi].tank_z[ti];
    let mesh = &m.meshes[b.mesh as usize];
    let (F, K, Ks) = (pl.omegas.len(), pl.rad.len(), col.len());
    let mut td = TankDrift { body: bi as i32, rho: m.env.density * b.tanks[ti].density, ..TankDrift::default() };
    let bx = read_f64(r, QuantityId::BodyPoints);
    let bn = read_f64(r, QuantityId::BodyPointsNormals);
    let ba = read_f64(r, QuantityId::BodyPointsAreas);
    let ids = read_i32(r, QuantityId::BodyPointsIds);
    let bp = read_f32(r, QuantityId::BodyPointsPressureRadiation);
    let bv = read_f32(r, QuantityId::BodyPointsVelocityRadiation);
    let Np = ba.len();
    // global axes (the tank problem's frame: the body's yaw) to body axes, its points about
    // the body origin (the tank problem places it at z = -Z_T)
    let c = (b.yaw * PI / 180.0).cos();
    let s = (b.yaw * PI / 180.0).sin();
    let loc = |v: &[f64]| v3(c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]);
    for q in 0..Np {
        let rel = [bx[3 * q] - b.position[0], bx[3 * q + 1] - b.position[1], bx[3 * q + 2] + zt];
        let xl = loc(&rel);
        td.x.push(xl);
        td.n.push(loc(&bn[3 * q..3 * q + 3]));
        td.area.push(ba[q]);
        td.zrel.push(xl[2] - zt);
    }
    // phi_j = g / w^2 (p / rho g) and grad phi_j = v / (i w) per unit velocity (the outputs
    // are per unit displacement), onto the body's columns
    td.phi = vec![Complex64::new(0.0, 0.0); F * Np * K];
    td.vel = vec![Complex64::new(0.0, 0.0); F * Np * K * 3];
    for f in 0..F {
        let w = pl.omegas[f];
        for q in 0..Np {
            for cs in 0..Ks {
                let os = (f * Np + q) * Ks + cs;
                let o = (f * Np + q) * K + col[cs];
                td.phi[o] = m.env.gravity / (w * w) * Complex64::new(bp[2 * os] as f64, bp[2 * os + 1] as f64);
                let mut vg = [Complex64::new(0.0, 0.0); 3];
                for e in 0..3 {
                    vg[e] = Complex64::new(bv[2 * (os * 3 + e)] as f64, bv[2 * (os * 3 + e) + 1] as f64)
                        / Complex64::new(0.0, w);
                }
                td.vel[o * 3] = c * vg[0] + s * vg[1];
                td.vel[o * 3 + 1] = -s * vg[0] + c * vg[1];
                td.vel[o * 3 + 2] = vg[2];
            }
        }
    }
    // the waterline: edges of the tank walls (every reflection) in the tank free surface,
    // with the point of their panel (ids: body, element of the tank problem, sx, sy)
    for q in 0..Np {
        let k = (ids[4 * q + 1] - 1) as usize;
        let sx = if ids[4 * q + 2] == 0 { 1.0 } else { ids[4 * q + 2] as f64 };
        let sy = if ids[4 * q + 3] == 0 { 1.0 } else { ids[4 * q + 3] as f64 };
        let p = &mesh.panels[recs[k] as usize];
        let tol = length_tol(p.diameter);
        for e in 0..4 {
            // the vertices as given (a warped panel's projection leaves the free surface)
            let a = p.raw[e];
            let cc = p.raw[(e + 1) % 4];
            if (a[2] - zt).abs() > tol || (cc[2] - zt).abs() > tol || !(norm(cc - a) > tol) {
                continue;
            }
            td.wl_pt.push(q as i32);
            td.wl_x.push(v3(sx * 0.5 * (a[0] + cc[0]), sy * 0.5 * (a[1] + cc[1]), zt));
            td.wl_n.push(td.n[q]);
            td.wl_dl.push(norm(cc - a));
        }
    }
    td
}

// The added mass and damping of tank ti of body bi (its tank problem's results r, plan sp,
// columns col) added to the body's: the rows of the body's rigid and generalized modes.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn add_tank_coefficients(m: &Problem, pl: &mut Plan, bi: usize, ti: usize, r: &Results, sp: &Plan, col: &[usize]) {
    let b = &m.bodies[bi];
    let (F, D, K) = (pl.omegas.len(), pl.D as usize, pl.rad.len());
    let (Ds, Ks) = (sp.D as usize, sp.rad.len());
    let off = m.derived()[bi].coordinate_offset as usize;
    let A = read_f32(r, QuantityId::AddedMass);
    let B = read_f32(r, QuantityId::RadiationDamping);
    let A0 = read_f32(r, QuantityId::AddedMassZeroFrequency);
    let Ai = read_f32(r, QuantityId::AddedMassInfiniteFrequency);
    // omega = 0: a mode with a net flux through the walls has no solution in the closed tank
    // (its potential, and with it every entry of its row and column, is undefined: NaN)
    let bad: Vec<bool> = if A0.is_empty() { Vec::new() } else { m.tank_zero_flux(bi, ti) };
    for i in 0..Ds.min(6 + b.generalized.len()) {
        for c in 0..Ks {
            // the added mass and the damping of the discrete tank problem (its damping, small away
            // from the sloshing modes, carries the response near them)
            if !A.is_empty() {
                for f in 0..F {
                    let o = (f * D + off + i) * K + col[c];
                    let os = (f * Ds + i) * Ks + c;
                    pl.tA[o] += A[os] as f64;
                    pl.tB[o] += B[os] as f64;
                }
            }
            let o = (off + i) * K + col[c];
            let os = i * Ks + c;
            if !A0.is_empty() {
                pl.tA0[o] += if bad[i] || bad[sp.rad[c].coord as usize] { f64::NAN } else { A0[os] as f64 };
            }
            if !Ai.is_empty() {
                pl.tAinf[o] += Ai[os] as f64;
            }
        }
    }
}

/// The field array of the exterior points, [F][Pe][n], over all field points, [F][P][n]:
/// the tank points from their tank problems (vel false: pressure, true: velocity; n = K or 3 K),
/// or zero (diffraction, rad false).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn field_all(m: &Problem, pl: &Plan, ext: Vec<Complex32>, F: i32, n: i32, rad: bool, vel: bool) -> Vec<Complex32> {
    if m.field_tanks.is_empty() {
        return ext; // the outputs' values, moved
    }
    let (F, n) = (F as usize, n as usize);
    let P = m.field_points.len();
    let mut Pe = 0;
    for q in 0..P {
        Pe += if m.exterior_point(q) { 1 } else { 0 };
    }
    let mut all = vec![Complex32::new(0.0, 0.0); F * P * n];
    for f in 0..F {
        let mut pe = 0;
        for q in 0..P {
            let dst = (f * P + q) * n;
            if m.exterior_point(q) {
                let src = (f * Pe + pe) * n;
                pe += 1;
                all[dst..dst + n].copy_from_slice(&ext[src..src + n]);
                continue;
            }
            if !rad {
                continue; // no incident or diffracted wave inside a tank
            }
            for tf in &pl.tfield {
                let v = if vel { &tf.vel } else { &tf.rad };
                if tf.point == q as i32 && !v.is_empty() {
                    all[dst..dst + n].copy_from_slice(&v[f * n..f * n + n]);
                }
            }
        }
    }
    all
}

/// Adds the pressure drift of the tank fluid of body bi at frequency fi, heading with the
/// motions Xr of every column (X: the six rigid motions of the body): o9 as PressureDrift::finish.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn add_tank_drift(
    pl: &Plan,
    bi: i32,
    fi: i32,
    w: f64,
    X: &[Complex64; 6],
    Rc: &Vec3,
    Xr: &[Complex64],
    o9: &mut [f64; 9],
    mut rec: Option<&mut QtfRecorder>,
) {
    let K = pl.rad.len();
    let iw = Complex64::new(0.0, w);
    let zc = Complex64::new(0.0, 0.0);
    for t in &pl.tdrift {
        if t.body != bi {
            continue;
        }
        // the tank fluid on its walls: the pressure expanded about the mean walls, z from the tank
        // free surface, no incident wave (theory §11.2 with the tank density)
        let mut pd = PressureDrift::new(X, Rc, t.rho, pl.grav, w, false);
        if let Some(r) = rec.as_deref_mut() {
            r.group(bi as usize, true, t.rho, false, Rc, 0.0, X);
        }
        let P = t.x.len();
        let at = |q: usize, ph: &mut Complex64, v: &mut [Complex64; 3]| {
            *ph = zc;
            *v = [zc; 3];
            for r in 0..K {
                let xr = iw * Xr[r];
                if xr == zc {
                    continue;
                }
                let o = (fi as usize * P + q) * K + r;
                *ph += xr * t.phi[o];
                for d in 0..3 {
                    v[d] += xr * t.vel[o * 3 + d];
                }
            }
        };
        for q in 0..P {
            let mut ph = zc;
            let mut v = [zc; 3];
            at(q, &mut ph, &mut v);
            pd.surface(ph, &v, &t.x[q], &t.n[q], t.zrel[q], t.area[q]);
            if let Some(r) = rec.as_deref_mut() {
                r.surface(ph, &v, &t.x[q], &t.n[q], t.zrel[q], t.area[q]);
            }
        }
        for e in 0..t.wl_pt.len() {
            let mut ph = zc;
            let mut v = [zc; 3];
            let q = t.wl_pt[e] as usize;
            at(q, &mut ph, &mut v);
            // the potential at the waterline point from that at the centroid of its panel
            let ph = crate::lo_velocity::at_waterline(ph, &v, &t.wl_x[e], &t.x[q]);
            pd.waterline(ph, &t.wl_x[e], &t.wl_n[e], t.wl_dl[e]);
            if let Some(r) = rec.as_deref_mut() {
                r.waterline(ph, &t.wl_x[e], &t.wl_n[e], t.wl_dl[e]);
            }
        }
        let zero = [0.0f64; 36];
        let mut o = [0.0f64; 9];
        pd.finish(0.0, &zero, &mut o);
        for k in 0..9 {
            o9[k] += o[k];
        }
    }
}

/// Add the tank terms to the forces of frequency fi (or of a limit) in its outputs `out` (of one
/// frequency).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn add_tank_terms(pl: &Plan, mode: FreqMode, fi: i32, out: &mut Output) {
    if pl.tA0.is_empty() {
        return; // no tanks (tA0 is sized whenever there are, tA only with ordinary frequencies)
    }
    let DK = pl.D as usize * pl.rad.len();
    if mode == FreqMode::Zero || mode == FreqMode::Infinite {
        let a = if mode == FreqMode::Zero { &mut out.A0 } else { &mut out.Ainf };
        let t = if mode == FreqMode::Zero { &pl.tA0 } else { &pl.tAinf };
        let mut k = 0;
        while k < DK && k < a.len() {
            a[k] += t[k] as f32;
            k += 1;
        }
        return;
    }
    let fi = fi as usize;
    for k in 0..DK {
        out.A[k] += pl.tA[fi * DK + k] as f32;
        out.B[k] += pl.tB[fi * DK + k] as f32;
    }
}
