//! The surfaces of a body (rule 3 of reference/inputs-rules §1): lids, pressure and interior
//! surfaces and free-surface plates in their planes, with their elements turned to the convention.

use crate::error::{Error, Location, Result, ResultExt, at_body};
use crate::geometry::{Placement, reflections};
use crate::model_codes::InteriorMethod;
use crate::problem::{Placed, Problem, all_records, body_label, model_place, placed_records, placement_of, records_of};
use crate::text::element_text;

impl Problem {
    // the elements of the pressure surfaces and free-surface plates with upward normals and of the
    // interior surfaces given as elements with downward ones, turned over (the meshes of the problem;
    // their z-normals are those of the body axes: the placement is a yaw and the reflections keep z).
    // Every element can be turned (Mesh::turn_over): these surfaces are planar, so their orientation
    // is a choice of the input, not of the geometry.
    fn orient_surfaces(&mut self) {
        let mut turn: Vec<(usize, Vec<i32>, bool)> = Vec::new(); // mesh, elements, downward wanted
        for b in &self.bodies {
            if b.interior_method == InteriorMethod::PanelRange {
                turn.push((b.mesh as usize, records_of(&b.interior_panels), false));
            }
            if b.interior_method == InteriorMethod::ExternalMesh {
                let mi = b.interior_mesh.expect("an external interior surface has its mesh (rule_bodies)");
                let im = &self.meshes[mi];
                turn.push((mi, all_records(im), false));
            }
            for p in &b.pressure {
                if let Some(pi) = p.mesh {
                    turn.push((pi, all_records(&self.meshes[pi]), true));
                } else {
                    turn.push((b.mesh as usize, records_of(&p.panels), true));
                }
            }
            // the plates of the sea and those of the tanks' free surfaces (normal -z into the fluid)
            for p in b.plates.iter().chain(b.tanks.iter().flat_map(|t| t.plates.iter())) {
                if let Some(pi) = p.mesh {
                    turn.push((pi, all_records(&self.meshes[pi]), true));
                } else {
                    turn.push((b.mesh as usize, records_of(&p.panels), true));
                }
            }
        }
        for (mi, recs, down) in turn {
            let m = &mut self.meshes[mi];
            for r in recs {
                let nz = m.shapes[r as usize].normal[2];
                if (down && nz > 0.5) || (!down && nz < -0.5) {
                    m.turn_over(r as usize);
                }
            }
        }
    }

    // Rule 3: lids at z = 0 with vertical normals (either orientation);
    // pressure surfaces horizontal, z <= 0, downward normals; interior surfaces at
    // z = 0 with upward normals; free-surface plates at z = 0 with downward normals. The elements of
    // a pressure or interior surface or a plate given the other way are turned over first (every
    // element: these surfaces are planar, so the flip is unambiguous; either orientation is
    // accepted, as for lids, whose column has no normal), so that the checks below see the
    // convention.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
    pub(super) fn rule_surfaces(&mut self) -> Result<()> {
        self.orient_surfaces();
        let at_z0 = |recs: &[Placed], what: &str, up: bool| {
            for r in recs {
                for h in &r.hull {
                    if h[2].abs() > r.tol {
                        return Err(Error::value(format!("{what}: a panel is not at z = 0")));
                    }
                }
                if up && !(r.normal[2] > 0.5) {
                    return Err(Error::value(format!("{what}: a panel normal does not point upwards")));
                }
                // an element with every vertex at z = 0 is flat in it
                if !up && !(r.normal[2].abs() > 0.5) {
                    return Err(Error::internal(format!("{what}: a panel normal is not vertical")));
                }
            }
            Ok(())
        };
        let at_z0_up = |recs: &[Placed], what: &str| at_z0(recs, what, true);
        for (li, l) in self.lids.iter().enumerate() {
            let at = format!("lids/{}", l.name);
            let lid = Location { lid: Some(li), ..Location::default() };
            if l.body >= 0 {
                let b = &self.bodies[l.body as usize];
                let m = &self.meshes[b.mesh as usize];
                at_z0(
                    &placed_records(m, &records_of(&l.panels), &placement_of(b), &reflections(m.isx, m.isy)),
                    &at,
                    false,
                )
                .located(lid)?;
            } else {
                let m = &self.meshes[l.mesh.expect("a lid of its own has its mesh (rule_lids)")];
                at_z0(
                    &placed_records(m, &all_records(m), &Placement::default(), &reflections(m.isx, m.isy)),
                    &at,
                    false,
                )
                .located(lid)?;
            }
        }
        for (bi, b) in self.bodies.iter().enumerate() {
            let here = at_body(bi as i32);
            let m = &self.meshes[b.mesh as usize];
            let place = placement_of(b);
            let at = body_label(b);
            if b.interior_method == InteriorMethod::PanelRange {
                at_z0_up(
                    &placed_records(m, &records_of(&b.interior_panels), &place, &reflections(m.isx, m.isy)),
                    &format!("{at} interior surface"),
                )
                .located(here)?;
            }
            if b.interior_method == InteriorMethod::ExternalMesh {
                let im =
                    &self.meshes[b.interior_mesh.expect("an external interior surface has its mesh (rule_bodies)")];
                at_z0_up(
                    &placed_records(im, &all_records(im), &place, &reflections(im.isx, im.isy)),
                    &format!("{at} interior surface"),
                )
                .located(here)?;
            }
            for (s, p) in b.pressure.iter().enumerate() {
                let ps = format!("{at} {}", element_text("pressure surface", model_place(p.place, s), &p.name));
                let recs = if let Some(pi) = p.mesh {
                    let pm = &self.meshes[pi];
                    placed_records(pm, &all_records(pm), &place, &reflections(pm.isx, pm.isy))
                } else {
                    placed_records(m, &records_of(&p.panels), &place, &reflections(m.isx, m.isy))
                };
                let zp = recs[0].hull[0][2];
                for r in &recs {
                    for h in &r.hull {
                        if (h[2] - zp).abs() > r.tol {
                            return Err(Error::value(format!("{ps}: the surface is not horizontal")).at(here));
                        }
                    }
                    if !(r.normal[2] < -0.5) {
                        return Err(Error::value(format!(
                            "{ps}: a panel normal does not point downwards (towards the water)"
                        ))
                        .at(here));
                    }
                }
                if zp > recs[0].tol {
                    return Err(Error::value(format!("{ps}: the surface lies above z = 0")).at(here));
                }
            }
            // free-surface plates: every element in z = 0 with a vertical normal (turned to -z above)
            for (s, p) in b.plates.iter().enumerate() {
                let ps = format!("{at} {}", element_text("free-surface plate", model_place(p.place, s), &p.name));
                let recs = if let Some(pi) = p.mesh {
                    let pm = &self.meshes[pi];
                    placed_records(pm, &all_records(pm), &place, &reflections(pm.isx, pm.isy))
                } else {
                    placed_records(m, &records_of(&p.panels), &place, &reflections(m.isx, m.isy))
                };
                for r in &recs {
                    if r.hull.iter().any(|h| h[2].abs() > r.tol) {
                        return Err(Error::value(format!(
                            "{ps}: an element does not lie in the free surface z = 0 (a free-surface plate lies in z = 0 \
                             with the body placed; a submerged plate is a thin sheet, a dipole)"
                        ))
                        .at(here));
                    }
                    if !(r.normal[2] < -0.5) {
                        // (every element is turned over above: a guard of the convention)
                        return Err(Error::value(format!(
                            "{ps}: a panel normal does not point downwards (into the water)"
                        ))
                        .at(here));
                    }
                }
            }
        }
        Ok(())
    }
}
