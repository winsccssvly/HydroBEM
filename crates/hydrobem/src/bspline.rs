//! B-spline basis functions and their first derivatives (the Cox-de Boor recursion): the knot span
//! and the basis with derivatives of Piegl & Tiller, The NURBS Book, 2nd ed., 1997, algorithms
//! A2.1 and A2.3 (the names ndu, left, right as there). The surfaces are evaluated from them by
//! mesh::patch_eval (rational ones too).

/// Index of the knot span containing u for order k (degree k-1) and n
/// coefficients: t[span] <= u < t[span+1], clamped at the right end.
pub fn find_span(t: &[f64], n: i32, k: i32, u: f64) -> i32 {
    let p = k - 1;
    if u >= t[n as usize] {
        return n - 1;
    }
    if u <= t[p as usize] {
        let mut s = p;
        while s + 1 < n && t[(s + 1) as usize] <= u {
            s += 1;
        }
        return s;
    }
    let (mut lo, mut hi) = (p, n);
    let mut mid = (lo + hi) / 2;
    while u < t[mid as usize] || u >= t[(mid + 1) as usize] {
        if u < t[mid as usize] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

/// Largest order of the evaluators (fixed arrays; geometry orders are checked
/// against MAX_SPLINE_ORDER of mesh.rs, potential orders are at most 6).
pub const MAX_BASIS_ORDER: i32 = 8;

const MB: usize = MAX_BASIS_ORDER as usize;

/// The k <= MAX_BASIS_ORDER non-zero basis functions N of order k on `span` at u
/// and, when dN is not None, their first derivatives (no allocation).
/// t: the knot vector, from the first knot of the basis on.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn basis(t: &[f64], k: i32, span: i32, u: f64, N: &mut [f64], dN: Option<&mut [f64]>) {
    let k = k as usize;
    let s = span as usize;
    let mut left = [0.0f64; MB];
    let mut right = [0.0f64; MB];
    let mut ndu = [[0.0f64; MB]; MB];
    ndu[0][0] = 1.0;
    for j in 1..k {
        left[j] = u - t[s + 1 - j];
        right[j] = t[s + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for j in 0..k {
        N[j] = ndu[j][k - 1];
    }
    let Some(dN) = dN else { return };
    // N'_{i,p} = p (N_{i,p-1}/(t_{i+p}-t_i) - N_{i+1,p-1}/(t_{i+p+1}-t_{i+1})), p = k - 1
    for r in 0..k {
        let mut d = 0.0;
        if k > 1 {
            if r >= 1 {
                d += ndu[r - 1][k - 2] / ndu[k - 1][r - 1];
            }
            if r + 2 <= k {
                d -= ndu[r][k - 2] / ndu[k - 1][r];
            }
        }
        dN[r] = (k - 1) as f64 * d;
    }
}

/// Values (d = 0) and first derivatives (d = 1) of the k non-zero basis
/// functions on `span` at u; out has 2 x k entries.
pub fn basis_and_derivative(t: &[f64], k: i32, span: i32, u: f64, out: &mut [f64]) {
    let (n, dn) = out.split_at_mut(k as usize);
    basis(t, k, span, u, n, Some(dn));
}
