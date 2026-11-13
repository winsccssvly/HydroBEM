//! The environment (rule 7 of reference/inputs-rules §1): depth, gravity and density, the domain
//! and the walls, and the field points (field_points.rs).

use crate::constants::fmt;
use crate::error::{Error, Location, Result, at_body};
use crate::geometry::{COINCIDENCE_TOL, Placement, Vec3, max_of, reflections};
use crate::model_codes::{Domain, InteriorMethod};
use crate::problem::{Problem, all_records, body_label, model_place, placed_records, placement_of};
use crate::text::element_text;

impl Problem {
    // the depth, gravity and density (rule 7)
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
    pub(super) fn rule_environment_values(&self) -> Result<()> {
        let env = self.env;
        if env.depth.is_nan() || !(env.depth > 0.0) {
            return Err(Error::value(format!("environment.depth must be > 0 or inf, got {}", fmt(env.depth))));
        }
        if !(env.gravity.is_finite() && env.gravity > 0.0) {
            return Err(Error::value(format!(
                "environment.gravity must be a positive finite number, got {}",
                fmt(env.gravity)
            )));
        }
        if !(env.density.is_finite() && env.density > 0.0) {
            return Err(Error::value(format!(
                "environment.density must be a positive finite number, got {}",
                fmt(env.density)
            )));
        }
        Ok(())
    }

    // Rule 7: environment ranges, domain consistency, wall sides, field points.
    pub(super) fn rule_environment_domain(&mut self) -> Result<()> {
        self.rule_environment_values()?;
        let env = self.env;
        let opt = self.opt;
        let channel = opt.domain == Domain::Channel;
        if channel {
            if !(opt.channel_width.is_finite() && opt.channel_width > 0.0) {
                return Err(Error::value("domain 'channel' requires a positive finite channel_width"));
            }
            if !env.depth.is_finite() {
                return Err(Error::value("domain 'channel' requires a finite depth"));
            }
            for &h in &self.headings {
                let mut r = h % 360.0;
                if r < 0.0 {
                    r += 360.0;
                }
                if r != 0.0 && r != 180.0 {
                    return Err(Error::value(format!(
                        "domain 'channel' allows headings 0 and 180 degrees only, got {}",
                        fmt(h)
                    )));
                }
            }
        } else if !opt.channel_width.is_nan() {
            return Err(Error::value("channel_width is only allowed with domain 'channel'"));
        }
        if !(opt.channel_reflect.is_finite() && opt.channel_reflect >= 0.0 && opt.channel_reflect <= 1.0) {
            return Err(Error::value("environment.channel_reflect must be in 0..1"));
        }
        if !channel && opt.channel_reflect != 1.0 {
            return Err(Error::value("channel_reflect is only allowed with domain 'channel'"));
        }

        // Wall sides: every point bounding a body surface (all elements, external
        // meshes, lids) must lie strictly inside the fluid.
        let inside = |p: &Vec3| -> bool {
            match opt.domain {
                Domain::XWall => p[0] > 0.0,
                Domain::YWall => p[1] > 0.0,
                Domain::OrthogonalWalls => p[0] > 0.0 && p[1] > 0.0,
                Domain::Channel => p[1].abs() < 0.5 * opt.channel_width,
                _ => true,
            }
        };
        let side = if opt.domain == Domain::XWall {
            "x > 0"
        } else if opt.domain == Domain::YWall {
            "y > 0"
        } else if opt.domain == Domain::OrthogonalWalls {
            "x > 0 and y > 0"
        } else {
            "|y| < W/2"
        };
        if opt.domain != Domain::OpenSea {
            // a body whose every element lies in a wall plane is a wavemaker in the wall (theory §10.3):
            // its elements are in the plane, not inside the fluid
            let wx = opt.domain == Domain::XWall || opt.domain == Domain::OrthogonalWalls;
            let wy = opt.domain == Domain::YWall || opt.domain == Domain::OrthogonalWalls;
            for bi in 0..self.bodies.len() {
                let b = &self.bodies[bi];
                let m = &self.meshes[b.mesh as usize];
                let mut wm = !m.shapes.is_empty();
                for r in placed_records(m, &all_records(m), &placement_of(b), &reflections(m.isx, m.isy)) {
                    let mut size: f64 = 0.0;
                    for h in &r.hull {
                        size = max_of(max_of(max_of(size, h[0].abs()), h[1].abs()), h[2].abs());
                    }
                    let tol = COINCIDENCE_TOL * max_of(1.0, size);
                    let (mut inx, mut iny) = (wx, wy);
                    for h in &r.hull {
                        inx = inx && h[0].abs() <= tol;
                        iny = iny && h[1].abs() <= tol;
                    }
                    wm = wm && (inx || iny);
                }
                self.derived_[bi].wavemaker = wm;
            }
            for bi in 0..self.bodies.len() {
                let here = at_body(bi as i32);
                let b = &self.bodies[bi];
                if self.derived_[bi].wavemaker {
                    continue;
                }
                let place = placement_of(b);
                let mut extra: Vec<Option<usize>> = vec![b.interior_mesh, b.dipole_mesh];
                for p in &b.pressure {
                    extra.push(p.mesh);
                }
                let mut ms: Vec<i32> = vec![b.mesh];
                for &e in &extra {
                    if let Some(i) = e
                        && (e != b.interior_mesh || b.interior_method == InteriorMethod::ExternalMesh)
                    {
                        ms.push(i as i32);
                    }
                }
                for &mi in &ms {
                    let m = &self.meshes[mi as usize];
                    for r in placed_records(m, &all_records(m), &place, &reflections(m.isx, m.isy)) {
                        for h in &r.hull {
                            if !inside(h) {
                                return Err(Error::value(format!(
                                    "{} is not strictly on the fluid side of the walls ({side})",
                                    body_label(b)
                                ))
                                .at(at_body(bi as i32))
                                .located(here));
                            }
                        }
                    }
                }
            }
            for (li, l) in self.lids.iter().enumerate() {
                if let Some(lm) = l.mesh {
                    let m = &self.meshes[lm];
                    for r in placed_records(m, &all_records(m), &Placement::default(), &reflections(m.isx, m.isy)) {
                        for h in &r.hull {
                            if !inside(h) {
                                return Err(Error::value(format!(
                                    "lids/{}: is not strictly on the fluid side of the walls ({side})",
                                    l.name
                                ))
                                .at(Location { lid: Some(li), ..Location::default() }));
                            }
                        }
                    }
                }
            }
            for (k, d) in self.fixed_dampers.iter().enumerate() {
                let m = &self.meshes[d.mesh as usize];
                for r in placed_records(m, &all_records(m), &Placement::default(), &reflections(m.isx, m.isy)) {
                    for h in &r.hull {
                        if !inside(h) {
                            return Err(Error::value(format!(
                                "{}: is not strictly on the fluid side of the walls ({side})",
                                element_text("fixed_dampers", model_place(d.place, k), &d.name)
                            )));
                        }
                    }
                }
            }
        }

        if self.has_field_points {
            self.check_field_points(&self.field_points, &self.field_tanks)?;
        }
        Ok(())
    }
}
