//! The higher order with surfaces of meshes of their own (reference/inputs-bodies §2, reference/inputs-model-objects §1): before the rules, the
//! other meshes of a body (its dipoles, pressure surfaces, free-surface plates and interior
//! surface, body coordinates)
//! and the damping lids given in global coordinates (moved into the coordinates of the first
//! body) are appended to the elements of the body's patch mesh, and their references become
//! element ranges of that mesh, the form the higher-order method solves. Flat patches with B-spline
//! ones become bilinear B-spline patches (the same surface); analytic patches join only analytic
//! ones.

use crate::error::{Error, Result};
use crate::gdf::analytic_desc;
use crate::geometry::{Vec3, v3};
use crate::mesh::{Mesh, Range, build_patch_mesh};
use crate::mesh_desc::{AnalyticPatchDesc, PatchDiscretization, PatchKind, PatchMeshDesc};
use crate::model_codes::InteriorMethod;
use crate::problem::{Joined, MeshOrigin, Problem};

/// One element of a patch mesh, in a form that can be concatenated.
#[derive(Clone, Debug)]
enum Rec {
    Flat([f64; 12]),
    Spline { ku: i32, kv: i32, mu: i32, mv: i32, knots: Vec<f64>, coef: Vec<f64>, weights: Option<Vec<f64>> },
    Analytic(AnalyticPatchDesc),
}

/// A rigid map of the plane: x' = R(-yaw) (x - t), the global coordinates into those of a body.
#[derive(Clone, Copy, Debug)]
struct ToBody {
    t: Vec3,
    c: f64,
    s: f64,
}

impl ToBody {
    fn identity() -> Self {
        ToBody { t: Vec3::ZERO, c: 1.0, s: 0.0 }
    }
    fn is_identity(&self) -> bool {
        self.t == Vec3::ZERO && self.c == 1.0 && self.s == 0.0
    }
    fn apply(&self, p: &[f64]) -> [f64; 3] {
        let (x, y, z) = (p[0] - self.t[0], p[1] - self.t[1], p[2] - self.t[2]);
        [self.c * x + self.s * y, -self.s * x + self.c * y, z]
    }
}

// the elements of a patch mesh, moved by the map (the element shapes of `Mesh` are rebuilt by
// build_patch_mesh)
fn records(m: &Mesh, map: &ToBody, at: &str) -> Result<Vec<(Rec, PatchDiscretization)>> {
    let mut out = Vec::new();
    let (mut ko, mut co) = (0usize, 0usize);
    for p in 0..m.count as usize {
        let d = m.discretization[p];
        let r = match m.geometry {
            PatchKind::Flat => {
                let mut v = [0.0; 12];
                for k in 0..4 {
                    v[3 * k..3 * k + 3].copy_from_slice(&map.apply(&m.vertices[12 * p + 3 * k..]));
                }
                Rec::Flat(v)
            }
            PatchKind::Spline => {
                let (ku, kv, mu, mv) = (m.orders[2 * p], m.orders[2 * p + 1], m.ncoef[2 * p], m.ncoef[2 * p + 1]);
                let nk = (mu + ku + mv + kv) as usize;
                let n = (mu * mv) as usize;
                let mut coef = Vec::with_capacity(3 * n);
                for k in 0..n {
                    coef.extend_from_slice(&map.apply(&m.coefficients[3 * (co + k)..]));
                }
                let weights = if m.weights.is_empty() { None } else { Some(m.weights[co..co + n].to_vec()) };
                let r = Rec::Spline { ku, kv, mu, mv, knots: m.knots[ko..ko + nk].to_vec(), coef, weights };
                ko += nk;
                co += n;
                r
            }
            PatchKind::Analytic => {
                if !map.is_identity() {
                    return Err(Error::value(format!(
                        "{at}an analytic mesh in global coordinates needs its body at the origin without yaw \
                         (the maps are not moved)"
                    )));
                }
                Rec::Analytic(analytic_desc(&m.analytic[p]))
            }
        };
        out.push((r, d));
    }
    Ok(out)
}

// a flat patch as the bilinear B-spline patch of the same surface: u and v exchanged, so that the
// normal into the fluid keeps its side (a flat patch has it along x_u x x_v, a spline patch
// against), the discretization with them
fn as_spline(v: &[f64; 12], d: &PatchDiscretization) -> (Rec, PatchDiscretization) {
    let p = |k: usize| [v[3 * k], v[3 * k + 1], v[3 * k + 2]];
    // the new u runs from vertex 0 to vertex 3, the new v from vertex 0 to vertex 1
    let coef = [p(0), p(3), p(1), p(2)].concat();
    let r = Rec::Spline {
        ku: 2,
        kv: 2,
        mu: 2,
        mv: 2,
        knots: vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        coef,
        weights: None,
    };
    let d2 = PatchDiscretization { nu: d.nv, nv: d.nu, ku: d.kv, kv: d.ku, iquo: d.iquo, iqui: d.iqui };
    (r, d2)
}

/// The patch mesh of the elements (base first), as built by build_patch_mesh.
fn merged(base: &Mesh, recs: Vec<(Rec, PatchDiscretization)>, at: &str) -> Result<Mesh> {
    let any_spline = recs.iter().any(|(r, _)| matches!(r, Rec::Spline { .. }));
    let any_flat = recs.iter().any(|(r, _)| matches!(r, Rec::Flat(_)));
    let any_analytic = recs.iter().any(|(r, _)| matches!(r, Rec::Analytic(_)));
    if any_analytic && (any_spline || any_flat) {
        return Err(Error::value(format!(
            "{at}analytic patches join only analytic ones in one body (the other meshes are flat or B-spline)"
        )));
    }
    let mut d = PatchMeshDesc {
        reference_length: base.reference_length,
        gravity: base.gravity,
        symmetry_x: base.isx,
        symmetry_y: base.isy,
        ..PatchMeshDesc::default()
    };
    d.geometry = if any_analytic {
        PatchKind::Analytic
    } else if any_spline {
        PatchKind::Spline
    } else {
        PatchKind::Flat
    };
    let rational = recs.iter().any(|(r, _)| matches!(r, Rec::Spline { weights: Some(_), .. }));
    for (r, disc) in recs {
        let (r, disc) = match (&r, d.geometry) {
            (Rec::Flat(v), PatchKind::Spline) => as_spline(v, &disc),
            _ => (r, disc),
        };
        match r {
            Rec::Flat(v) => d.vertices.extend_from_slice(&v),
            Rec::Spline { ku, kv, mu, mv, knots, coef, weights } => {
                d.spline_orders.extend_from_slice(&[ku, kv]);
                d.spline_counts.extend_from_slice(&[mu, mv]);
                d.knots.extend_from_slice(&knots);
                d.coefficients.extend_from_slice(&coef);
                if rational {
                    match weights {
                        Some(w) => d.weights.extend_from_slice(&w),
                        None => d.weights.extend(std::iter::repeat_n(1.0, (mu * mv) as usize)),
                    }
                }
            }
            Rec::Analytic(a) => d.analytic_patches.push(a),
        }
        d.discretization.push(disc);
        d.patch_count += 1;
    }
    d.knot_count = d.knots.len() as i64;
    d.coefficient_count = (d.coefficients.len() / 3) as i64;
    build_patch_mesh(&d).map_err(|e| Error::value(format!("{at}{}", e.message())))
}

/// The surfaces of meshes of their own of every body, and the lids in global coordinates, as
/// elements of the body's patch mesh (higher order only; nothing to do without them).
pub fn merge_other_meshes(m: &mut Problem) -> Result<()> {
    if !m.opt.higher_order || m.bodies.is_empty() {
        return Ok(());
    }
    for bi in 0..m.bodies.len() {
        let b = &m.bodies[bi];
        // the other meshes: (mesh, map, what) in the order appended
        enum What {
            Dipoles,
            Pressure(usize),
            Plate(usize),
            Interior,
            Lid(usize),
        }
        let mut other: Vec<(i32, ToBody, What)> = Vec::new();
        if let Some(dm) = b.dipole_mesh {
            other.push((dm as i32, ToBody::identity(), What::Dipoles));
        }
        for (s, p) in b.pressure.iter().enumerate() {
            if let Some(pi) = p.mesh {
                other.push((pi as i32, ToBody::identity(), What::Pressure(s)));
            }
        }
        for (s, p) in b.plates.iter().enumerate() {
            if let Some(pi) = p.mesh {
                other.push((pi as i32, ToBody::identity(), What::Plate(s)));
            }
        }
        if b.interior_method == InteriorMethod::ExternalMesh
            && let Some(im) = b.interior_mesh
        {
            other.push((im as i32, ToBody::identity(), What::Interior));
        }
        if bi == 0 {
            // a lid in global coordinates joins the first body, in its coordinates
            let yaw = b.yaw.rem_euclid(360.0).to_radians(); // whole turns exactly the identity
            let to_body = ToBody { t: v3(b.position[0], b.position[1], b.position[2]), c: yaw.cos(), s: yaw.sin() };
            for (li, l) in m.lids.iter().enumerate() {
                if l.body < 0
                    && let Some(lm) = l.mesh
                {
                    other.push((lm as i32, to_body, What::Lid(li)));
                }
            }
        }
        if other.is_empty() || b.mesh < 0 {
            continue;
        }
        let base = &m.meshes[b.mesh as usize];
        let at = format!("bodies/{}: ", b.name);
        if !base.patch {
            continue; // the rules refuse a low-order mesh in the higher order
        }
        if base.trimmed() {
            return Err(Error::value(format!(
                "{at}a trimmed body takes no surfaces of other meshes in the higher order (give them as \
                 elements of its mesh)"
            )));
        }
        let mut recs = records(base, &ToBody::identity(), &at)?;
        let mut ranges: Vec<Range> = Vec::new();
        let mut joined: Vec<Joined> = Vec::new();
        for (mi, map, what) in &other {
            let om = &m.meshes[*mi as usize];
            // a lid's messages are its own, naming the body it joins
            let (at, what_text) = match what {
                What::Dipoles => (at.clone(), "the mesh of its dipoles".to_string()),
                What::Pressure(s) => (at.clone(), format!("the mesh of pressure surface {}", s + 1)),
                What::Plate(s) => (at.clone(), format!("the mesh of free-surface plate {}", s + 1)),
                What::Interior => (at.clone(), "the mesh of its interior surface".to_string()),
                What::Lid(li) => (
                    format!("lids/{}: ", m.lids[*li].name),
                    format!("its mesh (joined to the first body '{}')", b.name),
                ),
            };
            if !om.patch || om.trimmed() {
                return Err(Error::value(format!("{at}{what_text} must be a flat, B-spline or analytic patch mesh")));
            }
            if om.isx != base.isx || om.isy != base.isy {
                return Err(Error::value(format!("{at}{what_text} must have the symmetry of the body's mesh")));
            }
            if matches!(what, What::Lid(_)) && (base.isx != 0 || base.isy != 0) {
                // a symmetric half or quadrant stays one in the body's coordinates only on its planes
                let (p, yaw) = (b.position, b.yaw);
                if yaw.rem_euclid(360.0) != 0.0 || (base.isx != 0 && p[0] != 0.0) || (base.isy != 0 && p[1] != 0.0) {
                    return Err(Error::value(format!(
                        "{at}{what_text} with symmetry needs the body on its symmetry planes without yaw"
                    )));
                }
            }
            let first = recs.len() as i32 + 1;
            recs.extend(records(om, map, &format!("{at}{what_text}: "))?);
            ranges.push(Range { first, last: recs.len() as i32 });
            let lid = if let What::Lid(li) = what { *li as i32 } else { -1 };
            joined.push(Joined { first: first - 1, count: recs.len() as i32 - first + 1, mesh: *mi, lid });
        }
        let base_count = base.count;
        let base_origin = m.origins.get(b.mesh as usize).map(|o| o.records.clone());
        let mesh = merged(base, recs, &at)?;
        // the stored elements of the body's elements, then of the appended ones those of their meshes
        let mut stored = base_origin.unwrap_or_else(|| (1..=base_count).collect());
        for j in &joined {
            stored.extend((0..j.count).map(|r| m.stored_record(j.mesh, r)));
        }
        let stored_count = m.origins.get(m.bodies[bi].mesh as usize).map_or(base_count, |o| o.stored);
        let index = m.meshes.len() as i32;
        m.meshes.push(mesh);
        m.origins.push(MeshOrigin { stored: stored_count, records: stored, joined, ..MeshOrigin::default() });
        let lid_body = bi as i32;
        let b = &mut m.bodies[bi];
        b.mesh = index;
        for ((_, _, what), r) in other.iter().zip(&ranges) {
            match what {
                What::Dipoles => {
                    b.dipole_panels.push(*r);
                    b.dipole_mesh = None;
                }
                What::Pressure(s) => {
                    b.pressure[*s].panels = vec![*r];
                    b.pressure[*s].mesh = None;
                }
                What::Plate(s) => {
                    b.plates[*s].panels = vec![*r];
                    b.plates[*s].mesh = None;
                }
                What::Interior => {
                    b.interior_method = InteriorMethod::PanelRange;
                    b.interior_panels = vec![*r];
                    b.interior_mesh = None;
                }
                What::Lid(li) => {
                    let l = &mut m.lids[*li];
                    l.body = lid_body;
                    l.panels = vec![*r];
                    l.mesh = None;
                }
            }
        }
    }
    Ok(())
}
