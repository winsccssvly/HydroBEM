//! The model rules (reference/inputs-rules §1), applied in a fixed order before any computation,
//! by area: the options (options.rs), the geometry (geometry.rs), the surfaces (surfaces.rs), the
//! frequencies (frequencies.rs), the environment (environment.rs) and the coordinates
//! (coordinates.rs); the plates in z = 0 (plate_rules.rs), the tanks' plates (tanks.rs) and the
//! field points (field_points.rs) beside them. One table (`RULES`) gives their order and the scopes
//! (`RuleScope`) each belongs to.

mod coordinates;
mod environment;
mod frequencies;
mod geometry;
mod options;
mod surfaces;

use crate::codes::{PointStatus, Warning};
use crate::error::{Diagnostic, Location, Result, at_body};
use crate::problem::{Problem, body_label, model_place};
use crate::quality::quality_warnings;
use crate::text::element_text;

pub(super) fn blank(s: &str) -> bool {
    s.chars().all(|c| c == ' ' || c == '\t' || c == '\r' || c == '\n')
}

pub(super) fn mode_name(j: i32) -> &'static str {
    const NAMES: [&str; 6] = ["surge", "sway", "heave", "roll", "pitch", "yaw"];
    if (1..=6).contains(&j) { NAMES[(j - 1) as usize] } else { "?" }
}

/// The problems the rules apply to: the model (a check, the route, a solve), a body's queries (and
/// those of the lids and fixed dampers), the free surface's queries, and the classification of
/// points. Each applies its subset of `RULES`, in the order of the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleScope {
    Model,
    Body,
    FreeSurface,
    Points,
}

use RuleScope::{Body, FreeSurface, Model, Points};

/// A rule of the table: its name (for the test of the orders), the scopes it belongs to and the
/// check.
struct Rule {
    #[cfg_attr(not(test), allow(dead_code))] // read by the test of the orders
    name: &'static str,
    scopes: &'static [RuleScope],
    check: fn(&mut Problem) -> Result<()>,
}

/// The rules in their one order; a scope applies those that name it, in this order, and the first
/// that fails is its refusal. The order of each scope is written out in the test below.
const RULES: &[Rule] = &[
    // the depth, gravity and density alone (the model and the free surface take them with the domain)
    Rule { name: "environment_values", scopes: &[Body], check: |p| p.rule_environment_values() },
    // the potential QTF's free surface (the model takes it with the options)
    Rule { name: "free_surface", scopes: &[FreeSurface], check: |p| p.rule_free_surface() },
    Rule { name: "options", scopes: &[Model, Points], check: |p| p.rule_options() },
    Rule { name: "geometry", scopes: &[Model, Body, FreeSurface, Points], check: |p| p.rule_geometry() },
    Rule { name: "surfaces", scopes: &[Model, Body, FreeSurface, Points], check: |p| p.rule_surfaces() },
    // the surfaces of z = 0 with plates, the sheets meeting them
    Rule { name: "plates", scopes: &[Model, FreeSurface], check: |p| crate::plate_rules::check(p) },
    // the plates of the tanks' free surfaces, in their problems
    Rule { name: "tank_plates", scopes: &[Model], check: |p| crate::tanks::check_tank_plates(p) },
    Rule { name: "something_to_solve", scopes: &[Model], check: |p| p.rule_something_to_solve() },
    Rule { name: "rao", scopes: &[Model], check: |p| p.rule_rao() },
    Rule { name: "frequencies", scopes: &[Model], check: |p| p.rule_frequencies() },
    Rule { name: "environment_domain", scopes: &[Model, FreeSurface], check: |p| p.rule_environment_domain() },
    Rule { name: "integration", scopes: &[Model], check: |p| p.rule_integration() },
    Rule { name: "zero_frequency_flux", scopes: &[Model], check: |p| p.rule_zero_frequency_flux() },
    Rule { name: "inertia", scopes: &[Model], check: |p| p.rule_inertia() },
];

impl Problem {
    /// The rules of a scope, in the order of the table; the first refusal ends them.
    fn apply_rules(&mut self, scope: RuleScope) -> Result<()> {
        for r in RULES.iter().filter(|r| r.scopes.contains(&scope)) {
            (r.check)(self)?;
        }
        Ok(())
    }

    /// Apply the model rules (reference/inputs-rules §1) in their fixed order (`RuleScope::Model`),
    /// then the warnings; cached until an input changes.
    /// Fails with ErrorKind::Value / ErrorKind::Type naming the broken rule.
    pub fn check_rules(&mut self) -> Result<()> {
        if self.rules_ok_ {
            return Ok(());
        }
        self.warnings.clear();
        self.apply_rules(Model)?;
        self.rule_warnings();
        self.rules_ok_ = true;
        Ok(())
    }

    /// The rules a query of a body or of the lids and fixed dampers needs (`RuleScope::Body`): the
    /// values of the environment, the geometry and the surfaces (not those of the options, outputs,
    /// frequencies and field points); nothing more when the model rules passed. The warnings of the
    /// last check are kept.
    pub fn check_body_rules(&mut self) -> Result<()> {
        if self.rules_ok_ {
            return Ok(());
        }
        let warnings = std::mem::take(&mut self.warnings);
        let r = self.apply_rules(Body);
        self.warnings = warnings;
        r
    }

    /// The rules of the free-surface queries (`RuleScope::FreeSurface`; the zones of the potential
    /// QTF's free surface, reference/queries): the free-surface items (`rule_free_surface`), the
    /// geometry and the surfaces of every body (their waterlines, the plates the free surface ends
    /// at), the plates in z = 0, the environment and its domain (the walls); not those of the
    /// options, the orders, the frequencies (`rule_frequency_values` where a zone's auto values take
    /// them) or the field points. The warnings stay those of the last full check.
    pub fn check_free_surface_rules(&mut self) -> Result<()> {
        if self.rules_ok_ {
            return Ok(());
        }
        let warnings = std::mem::take(&mut self.warnings);
        let r = self.apply_rules(FreeSurface);
        self.warnings = warnings;
        r
    }

    /// The rules that classify_field_points needs (`RuleScope::Points`: types, geometry, surfaces).
    pub fn check_geometry_rules(&mut self) -> Result<()> {
        self.warnings.clear();
        self.apply_rules(Points)
    }

    /// The rules of a scope, each with its own handling of the warnings: `check_rules`,
    /// `check_body_rules`, `check_free_surface_rules` or `check_geometry_rules`.
    pub fn check_scope(&mut self, scope: RuleScope) -> Result<()> {
        match scope {
            Model => self.check_rules(),
            Body => self.check_body_rules(),
            FreeSurface => self.check_free_surface_rules(),
            Points => self.check_geometry_rules(),
        }
    }

    // Warnings: valid inputs whose result the user may not expect; the mesh quality;
    // field points near a low-order panel.
    fn rule_warnings(&mut self) {
        let mut warnings = std::mem::take(&mut self.warnings);
        quality_warnings(self, &mut warnings);
        self.warnings = warnings;
        if self.has_field_points {
            let v = self.classify_field_points(&self.field_points, &self.field_tanks);
            for (i, p) in v.iter().enumerate() {
                if p.status != PointStatus::NearBody {
                    continue;
                }
                let at = Location { field_point: Some(i), ..at_body(p.index) };
                let message = format!(
                    "field point {} lies within a panel diameter of {}: its value has the error of the low-order method near a panel",
                    i + 1,
                    body_label(&self.bodies[p.index as usize])
                );
                self.warnings.push(Diagnostic::warning_at(Warning::FieldPointNearBody, at, message));
            }
        }
        if self.has_zero() && self.opt.radiation {
            for bi in 0..self.bodies.len() {
                for ti in 0..self.bodies[bi].tanks.len() {
                    if self.bodies[bi].tanks[ti].density == 0.0 {
                        continue;
                    }
                    let bad = self.tank_zero_flux(bi, ti);
                    let b = &self.bodies[bi];
                    let mut modes = String::new();
                    for &j in &b.modes {
                        if bad[(j - 1) as usize] {
                            modes += if modes.is_empty() { "" } else { ", " };
                            modes += mode_name(j);
                        }
                    }
                    for g in 0..b.generalized.len() {
                        if bad[6 + g] {
                            modes += if modes.is_empty() { "'" } else { ", '" };
                            modes += &b.generalized[g].name;
                            modes += "'";
                        }
                    }
                    if modes.is_empty() {
                        continue;
                    }
                    let message = format!(
                        "{}, {}: at omega = 0 the tank free surface is rigid and the {modes} mode has a net volume flux through the tank walls; its zero-frequency added mass is not defined and is NaN",
                        body_label(b),
                        element_text("tank", model_place(b.tanks[ti].place, ti), &b.tanks[ti].name)
                    );
                    self.warnings.push(Diagnostic::warning_at(Warning::TankZeroFrequency, at_body(bi as i32), message));
                }
            }
        }
        for bi in 0..self.bodies.len() {
            if self.opt.pressure_drift && !self.bodies[bi].pressure.is_empty() {
                // the pressure on the pressure surfaces is not integrated
                let message = format!(
                    "{}: second_order.drift_pressure_integration with pressure surfaces: the drift is integrated over the wetted surface only, without the pressure surfaces",
                    body_label(&self.bodies[bi])
                );
                self.warnings.push(Diagnostic::warning_at(
                    Warning::PressureDriftPressureSurfaces,
                    at_body(bi as i32),
                    message,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(scope: RuleScope) -> Vec<&'static str> {
        RULES.iter().filter(|r| r.scopes.contains(&scope)).map(|r| r.name).collect()
    }

    #[test]
    fn each_scope_applies_its_rules_in_this_order() {
        // a change of the table that reorders a scope, or moves a rule into or out of one, changes
        // which refusal a model with several faults gets: it shows here
        assert_eq!(
            order(Model),
            [
                "options",
                "geometry",
                "surfaces",
                "plates",
                "tank_plates",
                "something_to_solve",
                "rao",
                "frequencies",
                "environment_domain",
                "integration",
                "zero_frequency_flux",
                "inertia",
            ]
        );
        assert_eq!(order(Body), ["environment_values", "geometry", "surfaces"]);
        assert_eq!(order(FreeSurface), ["free_surface", "geometry", "surfaces", "plates", "environment_domain"]);
        assert_eq!(order(Points), ["options", "geometry", "surfaces"]);
    }

    #[test]
    fn every_rule_has_a_scope_and_a_name_of_its_own() {
        for (i, r) in RULES.iter().enumerate() {
            assert!(!r.scopes.is_empty(), "{}", r.name);
            assert!(RULES[..i].iter().all(|q| q.name != r.name), "{}", r.name);
        }
    }
}
