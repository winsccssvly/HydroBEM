//! The complex LU of either linear-algebra backend (linalg.rs) runs on the rayon pool installed by
//! its caller (the inner pool of a frequency), never on rayon's global pool, and its
//! solution does not depend on that pool's thread count. A test binary of its own: the global pool
//! is process-wide, and no other test may start it first.

use std::time::Instant;

use hydrobem_core::internals::linalg::lu_solve_c128;
use hydrobem_core::internals::parallel::pool;
use hydrobem_core::internals::progress::SolveContext;
use num_complex::Complex64;

#[test]
fn the_lu_runs_on_the_installed_pool() {
    let (n, nrhs) = (1200, 10);
    let mut a0 = vec![Complex64::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..n {
            let t = (i * 7 + j * 13) as f64;
            a0[i + j * n] = Complex64::new((0.3 * t).sin(), (0.17 * t).cos());
        }
        a0[j + j * n] += Complex64::new(3.0, 0.5);
    }
    let b0: Vec<Complex64> = (0..n * nrhs).map(|k| Complex64::new((k as f64).cos(), (0.5 * k as f64).sin())).collect();
    let ctx = SolveContext::default();
    let mut sols = Vec::new();
    for threads in [1, 3, 4] {
        let (mut a, mut b) = (a0.clone(), b0.clone());
        let p = pool("test", threads).unwrap();
        let t0 = Instant::now();
        p.install(|| lu_solve_c128(n as i32, &mut a, nrhs as i32, &mut b, ctx.statics())).unwrap();
        println!("LU n = {n}, {nrhs} right-hand sides, {threads} threads: {:.3} s", t0.elapsed().as_secs_f64());
        sols.push(b);
    }
    assert!(sols[1] == sols[0] && sols[2] == sols[0], "the solution depends on the thread count");
    // the global pool was never started: it can still be built
    assert!(rayon::ThreadPoolBuilder::new().num_threads(1).build_global().is_ok(), "the LU used the global pool");
}
