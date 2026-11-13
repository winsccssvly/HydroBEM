//! The heading pairs of the drifts and of the QTFs (results-and-files): the pair (h, h) of every
//! heading for unidirectional, every pair h_a <= h_b in the heading order for bidirectional (the
//! other order is the conjugate of the difference coefficient, at [b, a] of the QTFs, and the sum
//! coefficient itself). The engine computes the coefficients of all heading pairs [..][H][H][n] and
//! keeps those of the list.

use num_complex::Complex64;

use crate::codes::QuantityId;
use crate::results::{Results, put};

/// The pairs (h_a, h_b) of the headings.
pub fn heading_pairs(h: usize, bidirectional: bool) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for a in 0..h {
        if bidirectional {
            for b in a..h {
                v.push((a, b));
            }
        } else {
            v.push((a, a));
        }
    }
    v
}

/// The coefficients of the pairs from those of every heading pair: [lead][H][H][n] to
/// [lead][P][n].
pub fn select_pairs(full: &[Complex64], lead: usize, h: usize, n: usize, pairs: &[(usize, usize)]) -> Vec<Complex64> {
    let mut out = Vec::with_capacity(lead * pairs.len() * n);
    for l in 0..lead {
        for &(a, b) in pairs {
            let at = ((l * h + a) * h + b) * n;
            out.extend_from_slice(&full[at..at + n]);
        }
    }
    out
}

/// The places of the rows of the second-order loads on the coordinate axis (reference/results
/// ): the six rigid rows of each body and the generalized forces of its generalized modes.
pub struct CoordinateRows {
    /// the zero-based coordinates of the rigid modes (6 B)
    pub rigid: Vec<usize>,
    /// the zero-based coordinates of the generalized modes (G, the bodies in order)
    pub generalized: Vec<usize>,
    /// the coordinates (D)
    pub d: usize,
}

impl CoordinateRows {
    pub fn of(m: &crate::problem::Problem) -> CoordinateRows {
        CoordinateRows {
            rigid: m.rigid_coordinates(),
            generalized: m.generalized_coordinates(),
            d: m.coordinates() as usize,
        }
    }

    /// The loads on the coordinate axis ([lead][D]) from the rigid rows ([lead][6 B]) and the
    /// generalized forces ([lead][G]; empty when not computed): NaN on the other coordinates (the
    /// pressure coordinates, the generalized ones without forces). The rigid rows themselves when
    /// the coordinates are the rigid ones (D = 6 B: the rigid coordinates are then 0 .. 6 B in order,
    /// Problem::rigid_coordinates).
    pub fn place(&self, rigid: Vec<Complex64>, generalized: &[Complex64]) -> Vec<Complex64> {
        let n6 = self.rigid.len();
        if self.d == n6 {
            debug_assert!(self.rigid.iter().enumerate().all(|(k, &c)| k == c), "the rigid coordinates in order");
            return rigid; // moved
        }
        let g = self.generalized.len();
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let lead = rigid.len().checked_div(n6).unwrap_or(0);
        let mut out = vec![nan; lead * self.d];
        for l in 0..lead {
            let row = &mut out[l * self.d..(l + 1) * self.d];
            for (k, &c) in self.rigid.iter().enumerate() {
                row[c] = rigid[l * n6 + k];
            }
            if !generalized.is_empty() {
                for (k, &c) in self.generalized.iter().enumerate() {
                    row[c] = generalized[l * g + k];
                }
            }
        }
        out
    }
}

/// The pairs as a quantity [P, 2] of the headings in degrees.
pub fn put_pairs(r: &mut Results, q: QuantityId, headings: &[f64], pairs: &[(usize, usize)]) {
    let v: Vec<f64> = pairs.iter().flat_map(|&(a, b)| [headings[a], headings[b]]).collect();
    put(r, q, vec![pairs.len() as i64, 2], &v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pairs_and_their_selection() {
        assert_eq!(heading_pairs(3, false), vec![(0, 0), (1, 1), (2, 2)]);
        assert_eq!(heading_pairs(3, true), vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]);
        // [2][2][2][1] with the value 100 l + 10 a + b
        let full: Vec<Complex64> =
            (0..8).map(|i| Complex64::new((100 * (i / 4) + 10 * ((i / 2) % 2) + i % 2) as f64, 0.0)).collect();
        let s = select_pairs(&full, 2, 2, 1, &heading_pairs(2, true));
        let re: Vec<f64> = s.iter().map(|z| z.re).collect();
        assert_eq!(re, vec![0.0, 1.0, 11.0, 100.0, 101.0, 111.0]);
    }

    #[test]
    fn the_loads_on_the_coordinate_axis() {
        // two bodies: the first with two generalized modes, the second with one pressure mode
        // (coordinates 0..6 rigid, 6, 7 generalized; 8..14 rigid, 14 pressure); two lead rows
        let rows = CoordinateRows { rigid: (0..6).chain(8..14).collect(), generalized: vec![6, 7], d: 15 };
        let c = |x: f64| Complex64::new(x, -x);
        let rigid: Vec<Complex64> = (0..24).map(|i| c(i as f64)).collect();
        let gf: Vec<Complex64> = (0..4).map(|i| c(100.0 + i as f64)).collect();
        let out = rows.place(rigid.clone(), &gf);
        assert_eq!(out.len(), 30);
        for l in 0..2 {
            let row = &out[15 * l..15 * (l + 1)];
            assert_eq!(&row[..6], &rigid[12 * l..12 * l + 6]);
            assert_eq!(&row[6..8], &gf[2 * l..2 * l + 2]);
            assert_eq!(&row[8..14], &rigid[12 * l + 6..12 * l + 12]);
            assert!(row[14].re.is_nan() && row[14].im.is_nan());
        }
        // without the generalized forces: NaN on their coordinates
        assert!(rows.place(rigid.clone(), &[])[6].re.is_nan());
        // only rigid coordinates: the rigid rows themselves
        let rigid_only = CoordinateRows { rigid: (0..12).collect(), generalized: Vec::new(), d: 12 };
        assert_eq!(rigid_only.place(rigid.clone(), &[]), rigid);
    }
}
