//! The coordinates of the bodies (rules 9 and 10 of reference/inputs-rules §1): the flux of the
//! zero-frequency modes in finite depth and in the tanks, the inertia of the free coordinates, and
//! the generalized modes that tear the surface open.

use super::mode_name;
use crate::constants::fmt;
use crate::error::{Error, Result, at_body};
use crate::geometry::{Reflection, Vec3, cross, dot, norm, reflections};
use crate::hydrostatics::{rigid_inertia, symmetric_eigen};
use crate::mesh::{Mesh, Range};
use crate::model_codes::{Domain, PressureShape};
use crate::problem::{Problem, Role, all_records, body_label, records_of};
use crate::quadrature::record_quadrature;

// Relative threshold for a non-zero net volume flux.
const FLUX_TOL: f64 = 1e-6;

impl Problem {
    // Rule 9: finite-depth (and Channel) zero-frequency modes with a net volume
    // flux have a divergent added mass and are rejected.
    pub(super) fn rule_zero_frequency_flux(&mut self) -> Result<()> {
        if !(self.has_zero() && self.opt.radiation) {
            return Ok(());
        }
        if self.env.depth.is_infinite() && self.opt.domain != Domain::Channel {
            return Ok(());
        }
        for bi in 0..self.bodies.len() {
            let at_b = at_body(bi as i32);
            let here = at_b;
            let b = &self.bodies[bi];
            let m = &self.meshes[b.mesh as usize];
            let mut flux = [0.0f64; 6];
            let mut scale = [0.0f64; 6];
            // the wetted elements and the free-surface plates (a heaving plate pushes water too)
            let mut add_flux = |mesh: &Mesh, rec: i32, r: &Reflection| {
                for q in record_quadrature(mesh, rec, r) {
                    let x = q.x - b.rotation_center;
                    let mom = cross(x, q.nds);
                    let nj = [q.nds[0], q.nds[1], q.nds[2], mom[0], mom[1], mom[2]];
                    for j in 0..6 {
                        flux[j] += nj[j];
                        scale[j] += nj[j].abs();
                    }
                }
            };
            for r in reflections(m.isx, m.isy) {
                for rec in 0..m.count {
                    let role = self.derived_[bi].role[rec as usize];
                    if role == Role::Wet || role == Role::Plate {
                        add_flux(m, rec, &r);
                    }
                }
            }
            for p in &b.plates {
                if let Some(pi) = p.mesh {
                    let pm = &self.meshes[pi];
                    for r in reflections(pm.isx, pm.isy) {
                        for rec in 0..pm.count {
                            add_flux(pm, rec, &r);
                        }
                    }
                }
            }
            for &j in &b.modes {
                let k = (j - 1) as usize;
                if flux[k].abs() > FLUX_TOL * scale[k] {
                    return Err(Error::value(format!(
                            "{}: the zero-frequency {} mode has a net volume flux; its added mass diverges in finite depth and is not defined",
                            body_label(b),
                            mode_name(j)
                        )).at(at_b).located(here));
                }
            }
            for p in &b.pressure {
                let pm = &self.meshes[p.mesh.unwrap_or(b.mesh as usize)];
                let recs = if p.mesh.is_some() { all_records(pm) } else { records_of(&p.panels) };
                for mode in &p.modes {
                    let (mut f, mut sc) = (0.0, 0.0);
                    for r in reflections(pm.isx, pm.isy) {
                        for &rec in &recs {
                            for q in record_quadrature(pm, rec, &r) {
                                let shape = if mode.shape == PressureShape::Uniform {
                                    1.0
                                } else if mode.shape == PressureShape::SignX {
                                    if q.x[0] > 0.0 {
                                        1.0
                                    } else if q.x[0] < 0.0 {
                                        -1.0
                                    } else {
                                        0.0
                                    }
                                } else if q.x[1] > 0.0 {
                                    1.0
                                } else if q.x[1] < 0.0 {
                                    -1.0
                                } else {
                                    0.0
                                };
                                f += shape * q.nds[2];
                                sc += q.nds[2].abs();
                            }
                        }
                    }
                    if f.abs() > FLUX_TOL * sc {
                        return Err(Error::value(format!(
                                "{}: the zero-frequency pressure mode '{}' has a net volume flux; its added mass diverges in finite depth and is not defined",
                                body_label(b),
                                mode.name
                            )).at(at_b).located(here));
                    }
                }
            }
        }
        Ok(())
    }

    /// The coordinates of body bi (6 + G, in body order) whose zero-frequency problem of its tank
    /// ti has no solution: a net volume flux through the tank walls, which the rigid free surface of
    /// omega = 0 cannot take (heave; roll and pitch of a tank off the axes).
    pub fn tank_zero_flux(&self, bi: usize, ti: usize) -> Vec<bool> {
        let b = &self.bodies[bi];
        let m = &self.meshes[b.mesh as usize];
        let n = 6 + b.generalized.len();
        let mut flux = vec![0.0f64; n];
        let mut scale = vec![0.0f64; n];
        let mut add = |mesh: &Mesh, rec: i32, user_rec: i32, r: &Reflection| {
            for q in record_quadrature(mesh, rec, r) {
                let mom = cross(q.x - b.rotation_center, q.nds);
                let mut nj = vec![q.nds[0], q.nds[1], q.nds[2], mom[0], mom[1], mom[2]];
                for g in &b.generalized {
                    nj.push(dot(g.displacement(&q.x, user_rec), q.nds));
                }
                for j in 0..n {
                    flux[j] += nj[j];
                    scale[j] += nj[j].abs();
                }
            }
        };
        for r in reflections(m.isx, m.isy) {
            for rec in 0..m.count {
                if self.derived_[bi].tank[rec as usize] == ti as i32 {
                    add(m, rec, rec + 1, &r);
                }
            }
        }
        // the plates of its free surface on meshes of their own (a generalized piece with ranges
        // names elements of the body mesh only: user element 0 is in none)
        for pi in b.tanks[ti].plates.iter().filter_map(|p| p.mesh) {
            let pm = &self.meshes[pi];
            for r in reflections(pm.isx, pm.isy) {
                for rec in 0..pm.count {
                    add(pm, rec, 0, &r);
                }
            }
        }
        (0..n).map(|j| flux[j].abs() > FLUX_TOL * scale[j]).collect()
    }

    // Rule 10 (reference/inputs-rules §1, theory §3.1): the inertia of the free coordinates of a body (the rigid
    // inertia on its free rigid modes and mode_mass on its generalized coordinates, with their
    // couplings) is positive semi-definite, and every massless direction of it moves water. The
    // motion equation needs no inverse of the inertia: a massless direction takes its inertia
    // from the added mass (a massless lid or piston); one that moves no water has neither added
    // mass nor damping at any frequency and is refused.
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x >= a)` also rejects NaN, unlike `x < a`
    pub(super) fn rule_inertia(&mut self) -> Result<()> {
        if !self.opt.rao {
            return Ok(()); // the inertia enters the motions only
        }
        // relative to the unit diagonal of the scaled inertia: an eigenvalue below -NEG is a
        // negative kinetic energy, one below NULL a massless direction
        const NEG: f64 = 1e-9;
        const NULL: f64 = 1e-12;
        for bi in 0..self.bodies.len() {
            let at_b = at_body(bi as i32);
            let here = at_b;
            let b = &self.bodies[bi];
            let G = b.generalized.len();
            let rigid = b.has_cg && b.mass.is_some() && b.has_inertia;
            if (b.modes.is_empty() || !rigid) && G == 0 {
                continue;
            }
            // the free coordinates of the body: index into 0..6 + G and its name
            let mut free: Vec<(usize, String)> = Vec::new();
            if rigid {
                free.extend(b.modes.iter().map(|&j| ((j - 1) as usize, mode_name(j).to_string())));
            }
            free.extend(
                b.generalized.iter().enumerate().map(|(k, g)| (6 + k, format!("generalized mode '{}'", g.name))),
            );
            let n6 = 6 + G;
            let mut full = vec![0.0f64; n6 * n6];
            if rigid {
                let m = self.hydrostatics(bi as i32).mass;
                if !(m >= 0.0 && m.is_finite()) {
                    return Err(Error::value(format!(
                        "{}: the mass in use ({}) is not a non-negative finite number",
                        body_label(b),
                        fmt(m)
                    ))
                    .at(at_b)
                    .located(here));
                }
                let M = rigid_inertia(b, m);
                for i in 0..6 {
                    for j in 0..6 {
                        full[i * n6 + j] = M[i * 6 + j];
                    }
                }
            }
            if !b.mode_mass.is_empty() {
                for (f, &v) in full.iter_mut().zip(&b.mode_mass) {
                    *f += v;
                }
            }
            let n = free.len();
            // the symmetric part (the kinetic energy) on the free coordinates
            let S = |i: usize, j: usize| 0.5 * (full[free[i].0 * n6 + free[j].0] + full[free[j].0 * n6 + free[i].0]);
            let names = |v: &[f64]| {
                let big = v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
                let parts: Vec<String> =
                    (0..n).filter(|&i| v[i].abs() >= 1e-3 * big).map(|i| free[i].1.clone()).collect();
                if parts.len() == 1 {
                    format!("the free coordinate {}", parts[0])
                } else {
                    format!("the combination of the free coordinates {}", parts.join(", "))
                }
            };
            let refuse = |what: String| Err(Error::value(format!("{}: {what}", body_label(b))).at(at_b).located(here));
            // the massless directions: a coordinate of zero diagonal (its whole row must then be zero),
            // and the null space of the inertia scaled to a unit diagonal on the others
            let mut null: Vec<Vec<f64>> = Vec::new();
            let pos: Vec<usize> = (0..n).filter(|&i| S(i, i) != 0.0).collect();
            for i in 0..n {
                let d = S(i, i);
                if !(d >= 0.0) {
                    return refuse(format!(
                        "the inertia of the free coordinates is not positive semi-definite (a negative mass of {})",
                        free[i].1
                    ));
                }
                if d == 0.0 {
                    if let Some(j) = (0..n).find(|&j| S(i, j) != 0.0) {
                        return refuse(format!(
                            "the inertia of the free coordinates is not positive semi-definite ({} has no mass but an inertia coupling with {})",
                            free[i].1, free[j].1
                        ));
                    }
                    let mut v = vec![0.0; n];
                    v[i] = 1.0;
                    null.push(v);
                }
            }
            let np = pos.len();
            if np > 0 {
                let s: Vec<f64> = pos.iter().map(|&i| S(i, i).sqrt()).collect();
                let mut T = vec![0.0f64; np * np];
                for a in 0..np {
                    for c in 0..np {
                        T[a * np + c] = S(pos[a], pos[c]) / (s[a] * s[c]);
                    }
                }
                let (lam, U) = symmetric_eigen(T, np);
                for k in 0..np {
                    let mut v = vec![0.0; n];
                    for a in 0..np {
                        v[pos[a]] = U[a * np + k] / s[a];
                    }
                    if lam[k] < -NEG {
                        return refuse(format!(
                            "the inertia of the free coordinates is not positive semi-definite (a negative kinetic energy in the direction of {})",
                            names(&v)
                        ));
                    }
                    if lam[k] <= NULL {
                        null.push(v);
                    }
                }
            }
            for v in &null {
                if !self.moves_water(bi, &free.iter().map(|f| f.0).collect::<Vec<_>>(), v) {
                    return refuse(format!(
                        "{} has no mass and moves no water (no added mass or damping at any frequency): the motion equation is singular; give it a mass or fix it",
                        names(v)
                    ));
                }
            }
        }
        Ok(())
    }

    /// Whether the motion v of body bi (amplitudes of its coordinates idx, in 0..6 + G) has a normal
    /// velocity on the surfaces of the body in the water: the wetted elements, the thin sheets, the
    /// free-surface plates, the walls of its tanks and the plates of their free surfaces (of the
    /// body mesh or of meshes of their own; relative to the sum of the moduli of its
    /// coordinates' normal velocities).
    fn moves_water(&self, bi: usize, idx: &[usize], v: &[f64]) -> bool {
        let b = &self.bodies[bi];
        let m = &self.meshes[b.mesh as usize];
        let (mut flux, mut scale) = (0.0f64, 0.0f64);
        let mut add = |mesh: &Mesh, rec: i32, user_rec: i32, r: &Reflection| {
            for q in record_quadrature(mesh, rec, r) {
                let mom = cross(q.x - b.rotation_center, q.nds);
                let rigid = [q.nds[0], q.nds[1], q.nds[2], mom[0], mom[1], mom[2]];
                let mut s = 0.0;
                for (k, &c) in idx.iter().enumerate() {
                    if v[k] == 0.0 {
                        continue;
                    }
                    let nk =
                        if c < 6 { rigid[c] } else { dot(b.generalized[c - 6].displacement(&q.x, user_rec), q.nds) };
                    s += v[k] * nk;
                    scale += (v[k] * nk).abs();
                }
                flux += s.abs();
            }
        };
        for r in reflections(m.isx, m.isy) {
            for rec in 0..m.count {
                let role = self.derived_[bi].role[rec as usize];
                if matches!(role, Role::Wet | Role::Dipole | Role::Plate | Role::Tank) {
                    add(m, rec, rec + 1, &r);
                }
            }
        }
        // the plates of meshes of their own: their elements numbered after the body mesh's and
        // those of the pressure surfaces' own meshes (a generalized piece with ranges names elements
        // of the body mesh only)
        let mut next = m.count;
        for pi in b.pressure.iter().filter_map(|p| p.mesh) {
            next += self.meshes[pi].count;
        }
        for pi in b.plates.iter().filter_map(|p| p.mesh) {
            let pm = &self.meshes[pi];
            for r in reflections(pm.isx, pm.isy) {
                for rec in 0..pm.count {
                    add(pm, rec, next + rec + 1, &r);
                }
            }
            next += pm.count;
        }
        // the plates of its tanks' free surfaces on meshes of their own: in no piece with ranges
        // (user element 0, as tank_zero_flux numbers them)
        for t in &b.tanks {
            for pi in t.plates.iter().filter_map(|p| p.mesh) {
                let pm = &self.meshes[pi];
                for r in reflections(pm.isx, pm.isy) {
                    for rec in 0..pm.count {
                        add(pm, rec, 0, &r);
                    }
                }
            }
        }
        scale > 0.0 && flux > FLUX_TOL * scale
    }

    /// The first vertex of the wetted elements of body bi's mesh (low order; not its interior
    /// surface, sheets, dampers or lids) where one of its generalized modes jumps between two
    /// elements of different pieces, the surface torn open: the mode, the vertex (body-local) and the
    /// two one-based elements; None when every mode is continuous there. A kink (a hinge) is
    /// continuous, and so is a vertical jump between a free-surface plate's element and a wetted
    /// one: the plate's edge sliding along the wall (a piston in its tube, theory §11.2).
    pub fn torn_mode(&self, bi: usize) -> Option<(usize, Vec3, i32, i32)> {
        let b = &self.bodies[bi];
        let mesh = &self.meshes[b.mesh as usize];
        if mesh.patch || b.generalized.iter().all(|g| g.pieces.iter().all(|p| p.panels.is_empty())) {
            return None;
        }
        let inside = |r: &[Range], rec: i32| r.iter().any(|x| rec >= x.first && rec <= x.last);
        let skip = |rec: i32| {
            inside(&b.interior_panels, rec)
                || inside(&b.dipole_panels, rec)
                || b.dampers.iter().any(|d| inside(&d.panels, rec))
                || self.lids.iter().any(|l| l.body == bi as i32 && inside(&l.panels, rec))
        };
        let tol = 1e-6 * mesh.reference_length.max(1e-300);
        let plate = |rec: i32| b.plates.iter().any(|p| p.mesh.is_none() && inside(&p.panels, rec));
        for (k, gm) in b.generalized.iter().enumerate() {
            // the pieces that select an element
            let pieces = |rec: i32| -> Vec<usize> {
                (0..gm.pieces.len())
                    .filter(|&p| gm.pieces[p].panels.is_empty() || inside(&gm.pieces[p].panels, rec))
                    .collect()
            };
            let mut scale = 0.0f64;
            let mut seen: std::collections::HashMap<[i64; 3], (i32, Vec<usize>, Vec3)> =
                std::collections::HashMap::new();
            let mut pairs = Vec::new();
            for (r, p) in mesh.panels.iter().enumerate() {
                let rec = r as i32 + 1;
                if skip(rec) {
                    continue;
                }
                let ps = pieces(rec);
                for v in &p.raw {
                    let d = gm.displacement(v, rec);
                    scale = scale.max(norm(d));
                    let key = [(v[0] / tol).round() as i64, (v[1] / tol).round() as i64, (v[2] / tol).round() as i64];
                    match seen.get(&key) {
                        Some((r0, p0, d0)) if *p0 != ps => {
                            let j = d - *d0;
                            // a plate sliding along a wall: the jump vertical, the plate on one side
                            let slide = plate(*r0) != plate(rec);
                            let jump = if slide { j[0].hypot(j[1]) } else { norm(j) };
                            pairs.push((*v, *r0, rec, jump))
                        }
                        Some(_) => {}
                        None => {
                            seen.insert(key, (rec, ps.clone(), d));
                        }
                    }
                }
            }
            if let Some(&(x, ra, rb, _)) = pairs.iter().find(|q| q.3 > 1e-6 * scale) {
                return Some((k, x, ra, rb));
            }
        }
        None
    }
}
