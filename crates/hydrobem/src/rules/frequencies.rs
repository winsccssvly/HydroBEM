//! The frequencies, headings and field-point inputs (rule 6 of reference/inputs-rules §1).

use crate::constants::fmt;
use crate::error::{Error, Result};
use crate::problem::{Problem, finite3};

impl Problem {
    // Rule 6: frequencies, headings and field points.
    pub(super) fn rule_frequencies(&mut self) -> Result<()> {
        if self.omegas.is_empty() && !self.hydrostatics_only() {
            return Err(Error::value("environment.omegas is empty"));
        }
        self.rule_frequency_values()?;
        let mut hs: Vec<f64> = Vec::new();
        for &h in &self.headings {
            if !h.is_finite() {
                return Err(Error::value("environment.headings must be finite"));
            }
            if hs.contains(&h) {
                return Err(Error::value(format!("environment.headings has the duplicate value {}", fmt(h))));
            }
            hs.push(h);
        }
        if (self.has_zero() || self.has_infinite()) && !(self.opt.radiation && self.radiation_columns() > 0) {
            return Err(Error::value("frequency limits (omega = 0 or inf) require radiation with at least one mode"));
        }
        // dampers: the jump (i epsilon / omega) psi_n has no limit at omega = 0 or inf
        let mut dampers = !self.fixed_dampers.is_empty();
        for b in &self.bodies {
            dampers = dampers || !b.dampers.is_empty();
        }
        if dampers && (self.has_zero() || self.has_infinite()) {
            return Err(Error::value("frequency limits (omega = 0 or inf) are not defined with dipole dampers"));
        }
        if dampers && self.opt.drift_angles != 0 {
            return Err(Error::value(
                "second_order.drift_momentum_conservation is not valid with dipole dampers (the dampers absorb energy)",
            ));
        }
        if self.opt.diffraction && self.ordinary_frequencies() == 0 {
            return Err(Error::value("diffraction requires at least one ordinary frequency"));
        }
        if self.opt.diffraction && self.headings.is_empty() {
            return Err(Error::value("diffraction requires at least one heading"));
        }
        if !self.opt.diffraction && !self.headings.is_empty() {
            return Err(Error::value(
                "environment.headings are given but first_order.diffraction = false (headings are used only by diffraction)",
            ));
        }
        if self.has_field_points {
            if self.field_points.is_empty() {
                return Err(Error::value("first_order.field_points must have at least one point"));
            }
            if self.ordinary_frequencies() == 0 {
                return Err(Error::value("field points require at least one ordinary frequency"));
            }
            for p in &self.field_points {
                if !finite3(p) {
                    return Err(Error::value("field points must be finite"));
                }
            }
            if !self.field_tanks.is_empty() && self.field_tanks.len() != self.field_points.len() {
                return Err(Error::value("the tanks of the field points must have one entry per field point"));
            }
        }
        Ok(())
    }

    // the values of the frequencies (rule 6): >= 0, resolved, distinct; also a rule of the
    // free-surface queries whose auto values the frequencies decide
    pub(crate) fn rule_frequency_values(&self) -> Result<()> {
        // std::set<double>: the values equal under == (the omegas are not NaN when inserted)
        let mut seen: Vec<f64> = Vec::new();
        for &w in &self.omegas {
            if w.is_nan() || w < 0.0 || w == f64::NEG_INFINITY {
                return Err(Error::value(format!(
                    "environment.omegas must be >= 0 (0 and +inf request the limits), got {}",
                    fmt(w)
                )));
            }
            // the wavenumber K = omega^2 / g of an ordinary frequency must be a positive number
            // (an omega below about 1e-154 or above 1e154 rad/s underflows or overflows it)
            let k = w * w / self.env.gravity;
            if w > 0.0 && w.is_finite() && !(k > 0.0 && k.is_finite()) {
                return Err(Error::value(format!(
                    "environment.omegas: {} is outside the frequencies the engine resolves (omega^2 / g is not a positive number)",
                    fmt(w)
                )));
            }
            if seen.contains(&w) {
                return Err(Error::value(format!("environment.omegas has the duplicate value {}", fmt(w))));
            }
            seen.push(w);
        }
        Ok(())
    }
}
