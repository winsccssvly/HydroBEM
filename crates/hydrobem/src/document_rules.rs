//! The checks of a single item value, applied when the item is set.

/// The rule an item value is checked by alone (document.rs, `check_rule`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ItemRule {
    #[default]
    None,
    /// a real number (any value)
    Real,
    /// finite
    Finite,
    /// finite and > 0
    Positive,
    /// > 0 or inf
    Depth,
    /// finite and >= 0 (integers: >= 0)
    Nonnegative,
    /// finite in 0..1
    Unit,
    /// finite and within the float32 range
    Float32,
    /// an integer in lo..hi
    IntRange,
    /// a non-blank name
    Name,
    /// unique mode names in the order surge .. yaw
    Modes,
    /// element ranges: (N, 2) integers, one-based, first <= last, without overlaps (sorted when set)
    Ranges,
    /// >= 0, not NaN
    Omegas,
    /// at least one point, finite
    Points,
    /// finite, a symmetric 3 x 3 when given as a matrix
    Gyration,
    /// per row nu, nv >= 1; ku, kv in 2..6; iquo, iqui >= 1
    Discretization,
    /// finite coefficients, integer powers in 0..8
    Terms,
}
