//! The environment hooks of the engine, for internal accuracy studies and tests only:
//! - HB_DIAG_DOUBLE, the diagnostic double-precision mode: set when the engine starts, the Rankine
//!   integrals use the exact closed forms at every distance, the logarithmic-singularity integrals
//!   use no 3 x 3 far-field rule, the infinite-depth wave kernel uses no direct 2D tiles, and the
//!   influence data, matrices and LU are double precision (results are still stored as float32);
//! - HB_VALIDATE_<name>, the validation hooks of the sensitivity studies of the fixed constants
//!   (`VALIDATION_CONSTANTS`, `validation_constant`);
//! - HB_WATERLINE_CENTROID, the diagnostic of the low-order waterline potential: set when the
//!   engine first reads it, the waterline elements of the low-order pressure drift (of the bodies,
//!   of their quadratic QTFs and of the tank fluid) take the potential at the centroid of the
//!   waterline panel, without its expansion to the waterline (`lo_velocity::at_waterline`; the
//!   method before that expansion), to compare the two;
//! - the test hooks HB_AVAILABLE_MEMORY (bytes, replaces the available memory of the memory check,
//!   memory.rs) and HB_FIELD_STATICS_LIMIT (bytes, the limit of the stored static rows of the field
//!   points, low_order.rs), and HB_PROFILE (phase times on stderr).
//!
//! A hook that is set is visible: `environment_hooks` lists each one, which a solve records in the
//! meta data of its results (`RunMeta::environment_hooks`), and a hook that changes the numbers
//! (HB_DIAG_DOUBLE, HB_VALIDATE_<name>, HB_WATERLINE_CENTROID) is also a warning of the solve
//! (`Warning::EnvironmentHook`).
//! A HB_VALIDATE_ variable of no known constant, or a hook whose value does not parse, refuses the
//! solve (ErrorKind::Value) instead of being ignored. The variables are read with `var_os` and
//! `vars_os`: a variable that is not Unicode never panics (one of the hooks that is not Unicode is
//! refused like a value that does not parse).

use std::sync::LazyLock;

use crate::error::{Error, Result};

/// Whether the environment variable `name` is set (any value, Unicode or not).
pub fn env_set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

/// The number of the environment variable `name` (its text trimmed): None when it is not set,
/// ErrorKind::Value when it is set but is not Unicode or does not parse as a T.
pub(crate) fn env_number<T: std::str::FromStr>(name: &str) -> Result<Option<T>> {
    let Some(v) = std::env::var_os(name) else {
        return Ok(None);
    };
    match v.to_str().and_then(|t| t.trim().parse::<T>().ok()) {
        Some(x) => Ok(Some(x)),
        None => Err(Error::value(format!(
            "the environment variable {name} = {:?} is not a number (unset it, or give a number)",
            v.to_string_lossy()
        ))),
    }
}

/// Read once, at the first use.
pub static DIAG_DOUBLE: LazyLock<bool> = LazyLock::new(|| env_set("HB_DIAG_DOUBLE"));

/// Whether the diagnostic double-precision mode is on (HB_DIAG_DOUBLE set when it was first read).
pub fn diag_double() -> bool {
    *DIAG_DOUBLE
}

/// Read once, at the first use.
static WATERLINE_CENTROID: LazyLock<bool> = LazyLock::new(|| env_set("HB_WATERLINE_CENTROID"));

/// Whether the low-order waterline potential is taken at the panel centroid, unexpanded
/// (HB_WATERLINE_CENTROID set when it was first read; diagnostic only).
pub(crate) fn waterline_centroid() -> bool {
    *WATERLINE_CENTROID
}

/// The fixed constants of the method that the validation hooks can vary, with their values (the
/// numbers of the theory documents): the second circle of the far-field fit (theory §12.9), the
/// highest order with a local term and the dependence cutoff of the fit; the stencil step
/// min(0.02/k_s, 1e-3 R) and the offset of the direct method's waterline points (§12.8); the length of
/// the outer integral (4 longest wavelengths, at least 10 R), its taper and Gauss panel width (§12.9);
/// the widening of the boxes of several bodies (§12.8); the Gauss points per waterline panel side of
/// the higher-order pressure drift (theory §11.4).
pub const VALIDATION_CONSTANTS: [(&str, f64); 12] = [
    ("FIT_RADIUS_RATIO", 1.1),
    ("FIT_LOCAL_ORDERS", 30.0),
    ("FIT_DEPENDENCE", 0.05),
    ("STENCIL_WAVENUMBER", 0.02),
    ("STENCIL_RADIUS", 1e-3),
    ("WATERLINE_OFFSET", 1e-6),
    ("OUTER_WAVELENGTHS", 4.0),
    ("OUTER_RADII", 10.0),
    ("OUTER_TAPER", 0.3),
    ("OUTER_PANEL", 0.25),
    ("FS_GAP_FACTOR", 0.45),
    ("WATERLINE_GAUSS", 6.0),
];

const VALIDATE: &str = "HB_VALIDATE_";

/// The validation hooks of the environment: the constants set (name, value, the variable's text)
/// and the refusals (a variable of no known constant, a value that is not a finite number).
#[derive(Debug, Default)]
struct Validation {
    values: Vec<(&'static str, f64, String)>,
    refused: Vec<String>,
}

fn validation_of(vars: impl Iterator<Item = (std::ffi::OsString, std::ffi::OsString)>) -> Validation {
    let mut out = Validation::default();
    for (k, v) in vars {
        let key = k.to_string_lossy();
        let Some(name) = key.strip_prefix(VALIDATE) else {
            continue;
        };
        let Some(&(known, _)) = VALIDATION_CONSTANTS.iter().find(|(c, _)| *c == name && k.to_str().is_some()) else {
            out.refused.push(format!(
                "{key}: no validation constant {name} (the constants: {})",
                VALIDATION_CONSTANTS.map(|(c, _)| c).join(", ")
            ));
            continue;
        };
        match v.to_str().and_then(|t| t.trim().parse::<f64>().ok()).filter(|x| x.is_finite()) {
            Some(x) => out.values.push((known, x, v.to_string_lossy().trim().to_string())),
            None => out.refused.push(format!("{key} = {:?} is not a finite number", v.to_string_lossy())),
        }
    }
    out.values.sort_by(|a, b| a.0.cmp(b.0));
    out.refused.sort();
    out
}

/// Read once, at the first use (the variables of the process environment).
static VALIDATION: LazyLock<Validation> = LazyLock::new(|| validation_of(std::env::vars_os()));

/// Validation hook, for the sensitivity studies of the fixed constants only
/// (validation/constants.py, docs/validation/cases/constants.md): the constant `name` of
/// `VALIDATION_CONSTANTS`, or the number of the environment variable `HB_VALIDATE_<name>` when it
/// is set when the engine first reads one. The
/// variables are read once per process, so a study varies a constant by one process per value. The
/// rule: a hook never changes a default; without the variables every constant is the method's and
/// the results are unchanged. A variable that is refused (`environment_hooks`) leaves every constant
/// the method's (the solve is refused). An unknown `name` is a programming error (panic).
pub fn validation_constant(name: &str) -> f64 {
    let value = VALIDATION_CONSTANTS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)
        .unwrap_or_else(|| panic!("no validation constant {name}"));
    let v = &*VALIDATION;
    if !v.refused.is_empty() {
        return value;
    }
    v.values.iter().find(|(k, _, _)| *k == name).map_or(value, |(_, x, _)| *x)
}

/// A hook of the environment that is set.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ActiveHook {
    /// the variable and its value, `NAME=value` (`NAME` alone for HB_DIAG_DOUBLE,
    /// HB_WATERLINE_CENTROID and HB_PROFILE, whose value is not read)
    pub text: String,
    /// whether it changes the numbers of the results (HB_DIAG_DOUBLE, HB_VALIDATE_<name>,
    /// HB_WATERLINE_CENTROID)
    pub changes_results: bool,
}

/// The hooks of the environment that are set, in a fixed order (HB_DIAG_DOUBLE, the
/// HB_VALIDATE_<name> by name, HB_WATERLINE_CENTROID, HB_AVAILABLE_MEMORY, HB_FIELD_STATICS_LIMIT,
/// HB_PROFILE): those read
/// once per process as they were read, the others as they are now. ErrorKind::Value for a
/// HB_VALIDATE_ variable of no known constant and for a hook whose value is not a number.
pub(crate) fn environment_hooks() -> Result<Vec<ActiveHook>> {
    let v = &*VALIDATION;
    if !v.refused.is_empty() {
        return Err(Error::value(format!(
            "the validation hooks of the environment are refused (unset them): {}",
            v.refused.join("; ")
        )));
    }
    let mut out = Vec::new();
    if diag_double() {
        out.push(ActiveHook { text: "HB_DIAG_DOUBLE".to_string(), changes_results: true });
    }
    for (name, _, text) in &v.values {
        out.push(ActiveHook { text: format!("{VALIDATE}{name}={text}"), changes_results: true });
    }
    if waterline_centroid() {
        out.push(ActiveHook { text: "HB_WATERLINE_CENTROID".to_string(), changes_results: true });
    }
    if let Some(x) = env_number::<u64>("HB_AVAILABLE_MEMORY")? {
        out.push(ActiveHook { text: format!("HB_AVAILABLE_MEMORY={x}"), changes_results: false });
    }
    if let Some(x) = env_number::<f64>("HB_FIELD_STATICS_LIMIT")? {
        if x.is_nan() {
            return Err(Error::value("the environment variable HB_FIELD_STATICS_LIMIT = \"NaN\" is not a number"));
        }
        out.push(ActiveHook { text: format!("HB_FIELD_STATICS_LIMIT={x}"), changes_results: false });
    }
    if env_set("HB_PROFILE") {
        out.push(ActiveHook { text: "HB_PROFILE".to_string(), changes_results: false });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn without_the_variables_every_hooked_constant_has_its_documented_value() {
        // the values of the theory documents (theory §12.8-§12.9, theory §11.4), written out again
        let documented = [
            ("FIT_RADIUS_RATIO", 1.1),
            ("FIT_LOCAL_ORDERS", 30.0),
            ("FIT_DEPENDENCE", 0.05),
            ("STENCIL_WAVENUMBER", 0.02),
            ("STENCIL_RADIUS", 1e-3),
            ("WATERLINE_OFFSET", 1e-6),
            ("OUTER_WAVELENGTHS", 4.0),
            ("OUTER_RADII", 10.0),
            ("OUTER_TAPER", 0.3),
            ("OUTER_PANEL", 0.25),
            ("FS_GAP_FACTOR", 0.45),
            ("WATERLINE_GAUSS", 6.0),
        ];
        assert_eq!(documented.len(), VALIDATION_CONSTANTS.len());
        for (name, value) in documented {
            assert!(
                std::env::var_os(format!("HB_VALIDATE_{name}")).is_none(),
                "HB_VALIDATE_{name} is set: the tests run without the validation hooks"
            );
            assert_eq!(validation_constant(name), value, "{name}");
        }
    }

    #[test]
    #[should_panic(expected = "no validation constant")]
    fn an_unknown_name_is_refused() {
        validation_constant("NOT_A_CONSTANT");
    }

    // the variables of the environment as validation_of reads them: a known constant with a number
    // is kept with its text; a variable of no known constant, a value that does not parse or is not
    // finite and a value that is not Unicode are refused, each named; other variables (and those
    // that are not Unicode) are passed over without a panic
    #[test]
    fn the_validation_variables_are_kept_or_refused() {
        let os = |s: &str| OsString::from(s);
        let v = validation_of(
            [(os("HB_VALIDATE_OUTER_TAPER"), os(" 0.4 ")), (os("PATH"), os("x")), (os("HB_VALIDATEX"), os("1"))]
                .into_iter(),
        );
        assert_eq!(v.values, vec![("OUTER_TAPER", 0.4, "0.4".to_string())]);
        assert!(v.refused.is_empty());
        let v = validation_of(
            [
                (os("HB_VALIDATE_NOT_A_CONSTANT"), os("1")),
                (os("HB_VALIDATE_OUTER_TAPER"), os("1,5")),
                (os("HB_VALIDATE_OUTER_RADII"), os("inf")),
            ]
            .into_iter(),
        );
        assert!(v.values.is_empty());
        assert_eq!(v.refused.len(), 3, "{:?}", v.refused);
        assert!(v.refused.iter().any(|t| t.contains("no validation constant NOT_A_CONSTANT")));
        assert!(v.refused.iter().any(|t| t.contains("HB_VALIDATE_OUTER_TAPER = \"1,5\" is not a finite number")));
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            // an unpaired surrogate: not Unicode
            let bad = OsString::from_wide(&[0xD800]);
            let v = validation_of(
                [(os("HB_VALIDATE_OUTER_TAPER"), bad.clone()), (bad.clone(), os("1")), (os("OTHER"), bad)].into_iter(),
            );
            assert!(v.values.is_empty());
            assert_eq!(v.refused.len(), 1, "{:?}", v.refused);
        }
    }

    // without the variables no hook is active (the tests run without them)
    #[test]
    fn no_hook_is_active_in_the_tests() {
        for k in
            ["HB_DIAG_DOUBLE", "HB_WATERLINE_CENTROID", "HB_AVAILABLE_MEMORY", "HB_FIELD_STATICS_LIMIT", "HB_PROFILE"]
        {
            if env_set(k) {
                return; // a run with a hook set on purpose
            }
        }
        assert_eq!(environment_hooks().unwrap(), Vec::new());
    }
}
