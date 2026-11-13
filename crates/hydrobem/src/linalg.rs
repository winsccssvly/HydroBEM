//! Dense linear algebra on faer (pure Rust; pinned in Cargo.toml): the complex LU with
//! partial pivoting of the systems, complex64 and complex128 (theory §13.1), blocked in panels whose
//! blocks run on the threads of the current rayon pool; a real LU factored once and solved
//! many times; the products C = A B. Everything works in place on the column-major vectors
//! of the callers. Why complex64 in the low order and complex128 in the higher order:
//! theory §7 (grounds of the precision) and §13.1.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{factor as lu_factor, solve as lu_solve};
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve;
use faer::perm::PermRef;
use faer::traits::ComplexField;
use faer::{Accum, MatMut, MatRef, Par};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

use crate::codes::Stage;
use crate::error::{Error, Result};
use crate::progress::Work;

/// The element types of the complex LU (faer's complex32 and complex64 are those of num_complex).
pub trait LuScalar: ComplexField + Copy + PartialEq + Default + Send + Sync {
    fn is_finite(&self) -> bool;
    fn one() -> Self;
    fn minus_one() -> Self;
}

impl LuScalar for Complex32 {
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }
    fn minus_one() -> Self {
        Complex32::new(-1.0, 0.0)
    }
}

impl LuScalar for Complex64 {
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn minus_one() -> Self {
        Complex64::new(-1.0, 0.0)
    }
}

// The block sizes of the factorization, fixed by n so that the factors do not depend on the thread
// count: panels of PANEL columns; the row swaps, the block row of U and the trailing update
// in column blocks of UPDATE columns, the update of a block in row blocks of ROWS rows; the solve
// in blocks of RHS right-hand sides. Why blocked rather than faer's recursive LU of the whole
// matrix (progress and cancel every panel, the engine's own parallelism, the same speed within
// 5 %) and why the row blocks (without them 8-22 % slower on four threads): theory §13.1, grounds.
const PANEL: usize = 128;
const UPDATE: usize = 256;
const ROWS: usize = 512;
const RHS: usize = 32;

// The pivots of a panel in LAPACK's form from faer's row permutation of it (perm: row i of the
// factored panel is row perm[i] of the panel before): ipiv[i] the row (of the panel) that the
// i-th step of the elimination swapped with row i. The swaps applied in order give the permutation.
fn transpositions(perm: &[usize], ipiv: &mut [usize]) {
    let m = perm.len();
    let mut row: Vec<usize> = (0..m).collect(); // the row of the panel at each position
    let mut at: Vec<usize> = (0..m).collect(); // the position of each row of the panel
    for (i, p) in ipiv.iter_mut().enumerate() {
        let k = at[perm[i]];
        *p = k;
        row.swap(i, k);
        at[row[i]] = i;
        at[row[k]] = k;
    }
}

// The row swaps of LAPACK's laswp on whole columns: in each column of `cols` (leading dimension
// lda), rows r0 + i and ipiv[i] exchanged, i in order.
fn swap_rows<T: Copy>(cols: &mut [T], lda: usize, r0: usize, ipiv: &[usize]) {
    for col in cols.chunks_mut(lda) {
        for (i, &p) in ipiv.iter().enumerate() {
            col.swap(r0 + i, p);
        }
    }
}

// The work arrays of the panel factorizations (one panel at a time): faer's row permutation of the
// panel and its inverse, the scratch of faer's LU.
struct PanelWork {
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    mem: MemBuffer,
}

// faer's LU of the panel k0..k0 + kb from the diagonal down (`panel` its kb columns of A, leading
// dimension n), sequential; ipiv (kb values) its pivots in rows of A.
fn factor_panel<T: LuScalar>(n: usize, k0: usize, panel: &mut [T], ipiv: &mut [usize], pw: &mut PanelWork) {
    let (m, kb) = (n - k0, ipiv.len());
    let p = MatMut::from_column_major_slice_with_stride_mut(&mut panel[k0..], m, kb, n);
    let (perm, perm_inv) = (&mut pw.perm[..m], &mut pw.perm_inv[..m]);
    lu_factor::lu_in_place(p, perm, perm_inv, Par::Seq, MemStack::new(&mut pw.mem), Default::default());
    transpositions(perm, ipiv);
    for p in ipiv.iter_mut() {
        *p += k0; // in rows of A
    }
}

// A block of whole columns right of the panel k0 (`cols`, leading dimension n): the panel's row
// swaps, the block row U12 = L11^-1 A12 and the trailing update A22 -= L21 U12, the last in row
// blocks of ROWS rows on the threads of the current pool; every kernel sequential.
fn update<T: LuScalar>(n: usize, k0: usize, panel: &[T], piv: &[usize], cols: &mut [T]) {
    let (m, kb, w) = (n - k0, piv.len(), cols.len() / n);
    swap_rows(cols, n, k0, piv);
    let (l11, l21) = MatRef::from_column_major_slice_with_stride(&panel[k0..], m, kb, n).split_at_row(kb);
    let c = MatMut::from_column_major_slice_with_stride_mut(&mut cols[k0..], m, w, n);
    let (mut a12, a22) = c.split_at_row_mut(kb);
    triangular_solve::solve_unit_lower_triangular_in_place(l11, a12.as_mut(), Par::Seq);
    let u12 = a12.as_ref();
    a22.par_row_chunks_mut(ROWS).zip(l21.par_row_chunks(ROWS)).for_each(|(c, l)| {
        matmul(c, Accum::Add, l, u12, T::minus_one(), Par::Seq);
    });
}

// The LU factors of A (n x n, column-major) in place (unit lower L and U; ipiv the 0-based rows of
// the swaps, LAPACK's layout), right-looking in panels of PANEL columns. After faer's LU of
// a panel, on the threads of the current pool: the columns of the next panel are updated and that
// panel is factored (look-ahead), while the other columns are updated in blocks of UPDATE columns
// (left of the panel only the row swaps). Every kernel is sequential inside its block, and the
// blocks are fixed by n: each value is computed by the same operations on any thread, so the
// factors do not depend on the thread count.
fn factor<T: LuScalar>(n: usize, a: &mut [T], ipiv: &mut [usize], work: Work<'_>) -> Result<()> {
    let panels = n.div_ceil(PANEL) as i32;
    let mut pw = PanelWork {
        perm: vec![0; n],
        perm_inv: vec![0; n],
        mem: MemBuffer::new(lu_factor::lu_in_place_scratch::<usize, T>(n, PANEL, Par::Seq, Default::default())),
    };
    work.check_cancel()?;
    let kb = PANEL.min(n);
    factor_panel(n, 0, &mut a[..kb * n], &mut ipiv[..kb], &mut pw);
    for k0 in (0..n).step_by(PANEL) {
        work.check_cancel()?; // and the progress, every panel
        work.report(Stage::Lu, (k0 / PANEL) as i32, panels);
        let kb = PANEL.min(n - k0);
        // the columns left of the panel, the (factored) panel, the next panel, the columns after it
        let (left, rest) = a[..n * n].split_at_mut(k0 * n);
        let (panel, right) = rest.split_at_mut(kb * n);
        let (next, after) = right.split_at_mut(PANEL.min(right.len() / n) * n);
        let (piv, piv_next) = ipiv[k0..].split_at_mut(kb);
        let (panel, piv) = (&*panel, &*piv);
        let (ahead, others) = rayon::join(
            || -> Result<()> {
                if !next.is_empty() {
                    update(n, k0, panel, piv, next);
                    let kn = next.len() / n;
                    factor_panel(n, k0 + kb, next, &mut piv_next[..kn], &mut pw);
                }
                Ok(())
            },
            || -> Result<()> {
                let block = UPDATE * n;
                let blocks = left
                    .par_chunks_mut(block)
                    .map(|c| (false, c))
                    .chain(after.par_chunks_mut(block).map(|c| (true, c)));
                blocks.try_for_each(|(is_right, cols)| -> Result<()> {
                    work.check_cancel()?;
                    // each block holds whole columns of A, no other item touches them
                    if is_right {
                        update(n, k0, panel, piv, cols);
                    } else {
                        swap_rows(cols, n, k0, piv);
                    }
                    Ok(())
                })
            },
        );
        ahead?;
        others?;
    }
    Ok(())
}

// X = A^-1 B with the factors: the row swaps, then L and U, in blocks of RHS right-hand sides on
// the threads of the current pool.
fn substitute<T: LuScalar>(n: usize, a: &[T], ipiv: &[usize], nrhs: usize, b: &mut [T], work: Work<'_>) -> Result<()> {
    let lu = MatRef::from_column_major_slice(&a[..n * n], n, n);
    b[..n * nrhs].par_chunks_mut(RHS * n).try_for_each(|bj| -> Result<()> {
        work.check_cancel()?;
        // the columns of bj belong to this item only
        swap_rows(bj, n, 0, ipiv);
        let w = bj.len() / n;
        let mut x = MatMut::from_column_major_slice_mut(bj, n, w);
        triangular_solve::solve_unit_lower_triangular_in_place(lu, x.as_mut(), Par::Seq);
        triangular_solve::solve_upper_triangular_in_place(lu, x.as_mut(), Par::Seq);
        Ok(())
    })
}

// The 1-based index of the first zero on the diagonal of U (0: none), the zero pivot that
// LAPACK's getrf reports as info.
fn first_zero_pivot<T: LuScalar>(lu: MatRef<'_, T>) -> usize {
    let zero = T::default();
    (0..lu.nrows()).find(|&i| lu[(i, i)] == zero).map_or(0, |i| i + 1)
}

// The LU solve (lu_solve): the blocked factorization and solve with the checks of the
// matrix, the zero pivots and the solution.
fn solve<T: LuScalar>(n: i32, a: &mut [T], nrhs: i32, b: &mut [T], work: Work<'_>) -> Result<()> {
    let mut ipiv = vec![0usize; n as usize];
    lu_factor(n, a, &mut ipiv, work)?;
    lu_substitute(n, a, &ipiv, nrhs, b, work)
}

/// The factorization of lu_solve alone, kept for more right-hand sides: a (n x n, column-major)
/// overwritten by the factors, ipiv (n) the row swaps; the checks of lu_solve (finite entries, no
/// zero pivot).
pub fn lu_factor<T: LuScalar>(n: i32, a: &mut [T], ipiv: &mut [usize], work: Work<'_>) -> Result<()> {
    let n = n as usize;
    assert!(a.len() >= n * n && ipiv.len() >= n);
    if a[..n * n].iter().any(|v| !v.is_finite()) {
        return Err(Error::numerical("non-finite system matrix"));
    }
    factor(n, a, &mut ipiv[..n], work)?;
    let info = first_zero_pivot(MatRef::from_column_major_slice(&a[..n * n], n, n));
    if info > 0 {
        return Err(Error::numerical(format!("singular system matrix (zero pivot {info})")));
    }
    Ok(())
}

/// The substitution of lu_solve with the factors of lu_factor: b (n x nrhs, column-major)
/// overwritten by X; the check of the solution.
pub fn lu_substitute<T: LuScalar>(
    n: i32,
    a: &[T],
    ipiv: &[usize],
    nrhs: i32,
    b: &mut [T],
    work: Work<'_>,
) -> Result<()> {
    let (n, nrhs) = (n as usize, nrhs as usize);
    assert!(a.len() >= n * n && b.len() >= n * nrhs);
    substitute(n, a, &ipiv[..n], nrhs, b, work)?;
    if b[..n * nrhs].iter().any(|v| !v.is_finite()) {
        return Err(Error::numerical("non-finite solution"));
    }
    Ok(())
}

/// A real square matrix factored once and solved for many right-hand sides.
#[derive(Clone, Debug, Default)]
pub struct RealLu {
    pub n: i32,
    pub lu: Vec<f64>, // column-major n x n, the factors
    pub perm: Vec<usize>,
    pub perm_inv: Vec<usize>,
}

impl RealLu {
    /// ErrorKind::Numerical for a singular matrix.
    pub fn factor(&mut self, size: i32, a: Vec<f64>) -> Result<()> {
        let n = size as usize;
        self.n = size;
        self.lu = a;
        self.perm = vec![0; n];
        self.perm_inv = vec![0; n];
        // the checks of the complex solve (lu_solve): finite entries, no zero pivot
        if self.lu.iter().any(|v| !v.is_finite()) {
            return Err(Error::numerical("non-finite matrix"));
        }
        let mut lu = MatMut::from_column_major_slice_mut(&mut self.lu[..n * n], n, n);
        let mut mem = MemBuffer::new(lu_factor::lu_in_place_scratch::<usize, f64>(n, n, Par::Seq, Default::default()));
        lu_factor::lu_in_place(
            lu.as_mut(),
            &mut self.perm,
            &mut self.perm_inv,
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let zero = (0..n).find(|&i| lu[(i, i)] == 0.0).map_or(0, |i| i + 1);
        if zero > 0 {
            return Err(Error::numerical(format!("singular matrix (zero pivot {zero})")));
        }
        Ok(())
    }

    /// b (n complex values) overwritten by the solution of A x = b.
    pub fn solve(&self, b: &mut [Complex64]) -> Result<()> {
        let n = self.n as usize;
        // the real and the imaginary parts as two right-hand sides of the real factors
        let mut x = vec![0.0; 2 * n];
        for k in 0..n {
            x[k] = b[k].re;
            x[n + k] = b[k].im;
        }
        let lu = MatRef::from_column_major_slice(&self.lu[..n * n], n, n);
        let perm = PermRef::new_checked(&self.perm, &self.perm_inv, n);
        let mut mem = MemBuffer::new(lu_solve::solve_in_place_scratch::<usize, f64>(n, 2, Par::Seq));
        let rhs = MatMut::from_column_major_slice_mut(&mut x, n, 2);
        lu_solve::solve_in_place(lu, lu, perm, rhs, Par::Seq, MemStack::new(&mut mem));
        for k in 0..n {
            if !x[k].is_finite() || !x[n + k].is_finite() {
                return Err(Error::numerical("non-finite solution"));
            }
            b[k] = Complex64::new(x[k], x[n + k]);
        }
        Ok(())
    }
}

/// Solve A X = B in place with an LU factorization with partial pivoting.
/// a: n x n, column-major, overwritten by the factors; b: n x nrhs, column-major, overwritten by X;
/// The blocks run on the threads of the current pool (the result does not depend on their count);
/// work: the cancel check and the progress of the factorization (every panel).
/// ErrorKind::Numerical for singular or non-finite systems.
pub fn lu_solve_c64(n: i32, a: &mut [Complex32], nrhs: i32, b: &mut [Complex32], work: Work<'_>) -> Result<()> {
    solve(n, a, nrhs, b, work)
}

/// The same in double precision (the higher order and the diagnostic mode).
pub fn lu_solve_c128(n: i32, a: &mut [Complex64], nrhs: i32, b: &mut [Complex64], work: Work<'_>) -> Result<()> {
    solve(n, a, nrhs, b, work)
}

/// The LU solve of either precision (the generic code of the solvers).
pub fn lu_solve<T: LuScalar>(n: i32, a: &mut [T], nrhs: i32, b: &mut [T], work: Work<'_>) -> Result<()> {
    solve(n, a, nrhs, b, work)
}

/// C = A B (no transposes, column-major), either precision: m x k times k x n into m x n; a, b and
/// c hold the matrices with the leading dimensions lda, ldb and ldc. Sequential: the callers run it
/// inside their own parallel items.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
pub fn gemm<T: LuScalar>(
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    c: &mut [T],
    ldc: usize,
) {
    let a = MatRef::from_column_major_slice_with_stride(a, m, k, lda);
    let b = MatRef::from_column_major_slice_with_stride(b, k, n, ldb);
    let c = MatMut::from_column_major_slice_with_stride_mut(c, m, n, ldc);
    matmul(c, Accum::Replace, a, b, T::one(), Par::Seq);
}

/// The bytes that faer's matrix products keep on each thread that has run one, until the thread
/// ends (the memory estimate): the packing buffers of the two operands of the x86
/// kernels (private-gemm-x86, thread-local), each the size of the L3 cache in whole pages, the
/// cache sizes as those kernels take them (CPUID; at least 32 KiB, 256 KiB and 2 MiB, each level at
/// least four times the one below).
pub fn thread_workspace_bytes() -> i64 {
    let c = cache_bytes().unwrap_or([32 << 10, 256 << 10, 2 << 20]);
    let l1 = c[0].max(32 << 10);
    let l2 = c[1].max(256 << 10).max(4 * l1);
    let l3 = c[2].max(2 << 20).max(4 * l2);
    (2 * l3.div_ceil(4096) * 4096) as i64
}

/// The bytes of the buffers of thread_workspace_bytes that a thread updating the trailing blocks
/// of the LU of an n x n matrix of s-byte elements writes (resident; the rest of the buffers is
/// reserved, not touched): the products A22 -= L21 U12 pack a row block of L21 (ROWS x PANEL, the
/// rows rounded up to the register rows of the kernels, at most 16 more) and U12 (PANEL x UPDATE,
/// a few columns more); at most the buffers.
pub fn update_resident_bytes(n: i64, s: i64) -> i64 {
    let (rows, panel, update) = ((ROWS as i64).min(n), (PANEL as i64).min(n), (UPDATE as i64).min(n));
    let b = ((rows + 16) * panel + (update + 4) * panel) * s;
    b.min(thread_workspace_bytes())
}

/// The bytes of those buffers that the thread factoring the panels of the LU writes: the recursive
/// products of faer's panel LU pack about n x PANEL/2 elements; at most the buffers.
pub fn panel_resident_bytes(n: i64, s: i64) -> i64 {
    (n * (PANEL as i64 / 2).min(n) * s).min(thread_workspace_bytes())
}

// The data cache sizes in bytes (L1, L2, L3) from CPUID: Intel's deterministic cache parameters,
// AMD's cache information; None for another vendor.
fn cache_bytes() -> Option<[usize; 3]> {
    use raw_cpuid::{CacheType, CpuId};
    let cpuid = CpuId::new();
    match cpuid.get_vendor_info()?.as_str() {
        "GenuineIntel" => {
            let mut c = [0usize; 3];
            for p in cpuid.get_cache_parameters()? {
                let level = p.level() as usize;
                if matches!(p.cache_type(), CacheType::Data | CacheType::Unified) && (1..=3).contains(&level) {
                    c[level - 1] = p.associativity() * p.sets() * p.coherency_line_size();
                }
            }
            Some(c)
        }
        "AuthenticAMD" => {
            let l1 = cpuid.get_l1_cache_and_tlb_info()?;
            let l23 = cpuid.get_l2_l3_cache_and_tlb_info()?;
            // L1 and L2 in KiB, L3 in units of 512 KiB
            let kib = 1024;
            Some([
                l1.dcache_size() as usize * kib,
                l23.l2cache_size() as usize * kib,
                l23.l3cache_size() as usize * 512 * kib,
            ])
        }
        _ => None,
    }
}

/// The linear-algebra library (the build information): the version pinned in Cargo.toml.
pub fn version() -> String {
    "faer 0.24.4 (pure Rust)".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::progress::SolveContext;

    // a diagonally dominant test matrix of order n (column-major) and x = 1 + i k / n
    fn system(n: usize) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                let t = (i * 7 + j * 13) as f64;
                a[i + j * n] = Complex64::new((0.3 * t).sin(), (0.17 * t).cos()) * 0.1;
            }
            a[j + j * n] += Complex64::new(n as f64 * 0.2 + 1.0, 0.5);
        }
        let x: Vec<Complex64> = (0..n).map(|k| Complex64::new(1.0, k as f64 / n as f64)).collect();
        let mut b = vec![Complex64::new(0.0, 0.0); n];
        for j in 0..n {
            for i in 0..n {
                b[i] += a[i + j * n] * x[j];
            }
        }
        (a, b, x)
    }

    #[test]
    fn complex128_blocked_lu_solves_and_is_thread_independent() {
        let n = 300; // three panels, two update blocks
        let (a0, b0, x) = system(n);
        let mut sols = Vec::new();
        let ctx = SolveContext::default();
        for threads in [1, 4] {
            let (mut a, mut b) = (a0.clone(), b0.clone());
            let pool = crate::parallel::pool("test", threads).unwrap();
            pool.install(|| lu_solve_c128(n as i32, &mut a, 1, &mut b, ctx.statics())).unwrap();
            let err = b.iter().zip(&x).map(|(p, q)| (p - q).norm()).fold(0.0, f64::max);
            assert!(err < 1e-10, "error {err}");
            sols.push(b);
        }
        assert_eq!(sols[0], sols[1], "the solution depends on the thread count");
    }

    #[test]
    fn complex64_lu_solves() {
        let n = 200;
        let (a0, b0, x) = system(n);
        let mut a: Vec<Complex32> = a0.iter().map(|v| Complex32::new(v.re as f32, v.im as f32)).collect();
        let mut b: Vec<Complex32> = b0.iter().map(|v| Complex32::new(v.re as f32, v.im as f32)).collect();
        lu_solve_c64(n as i32, &mut a, 1, &mut b, SolveContext::default().statics()).unwrap();
        let err = b.iter().zip(&x).map(|(p, q)| (p.re as f64 - q.re).hypot(p.im as f64 - q.im)).fold(0.0, f64::max);
        assert!(err < 1e-4, "error {err}");
    }

    #[test]
    fn singular_and_real_lu() {
        let mut a = vec![Complex64::new(0.0, 0.0); 4];
        let mut b = vec![Complex64::new(1.0, 0.0); 2];
        let e = lu_solve_c128(2, &mut a, 1, &mut b, SolveContext::default().statics()).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Numerical);

        let mut lu = RealLu::default();
        lu.factor(2, vec![2.0, 1.0, 1.0, 3.0]).unwrap(); // [[2, 1], [1, 3]] column-major
        let mut x = vec![Complex64::new(3.0, 5.0), Complex64::new(4.0, 10.0)];
        lu.solve(&mut x).unwrap();
        assert!((x[0] - Complex64::new(1.0, 1.0)).norm() < 1e-14 && (x[1] - Complex64::new(1.0, 3.0)).norm() < 1e-14);
    }
}
