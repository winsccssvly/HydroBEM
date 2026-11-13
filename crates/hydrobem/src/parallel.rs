//! The threads of a solve: rayon pools sized by the options, built per solve. The
//! statics run on a pool of frequency_threads x inner_threads threads; the frequencies on
//! frequency_threads workers, each with an inner pool of inner_threads threads of its own, on
//! which the parallel loops inside its frequency (assembly, LU, outputs) run. An item computes the
//! same values on any thread and the items write disjoint data (chunks of the outputs, or owned
//! results written after), so a result does not depend on the thread count.
//!
//! The nesting: a frequency worker takes the next frequency and installs it on its own inner pool,
//! where every parallel loop of the frequency runs. The frequencies are taken by a broadcast to the
//! workers (one job per worker that loops over the frequencies), so a worker waiting for its inner
//! pool finds no other frequency job to steal: at most frequency_threads frequencies (each with
//! its workspace, counted by the memory estimate) are in progress at once, and a pool never waits on
//! itself. A worker keeps its number (0..frequency_threads − 1, rayon's broadcast index) for the
//! whole loop: the progress board shows it on every item it runs. The second-order steps that run
//! by item use the same workers (`StepRows`, `FrequencyPools::for_each_item`), so that a worker
//! number means the same thread in every phase.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::progress::{ItemWork, StepWork};

use rayon::iter::Either;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{Error, Result};
use crate::progress::SolveContext;

/// A pool of `threads` threads (at least one) named `name`-i.
pub fn pool(name: &'static str, threads: i32) -> Result<ThreadPool> {
    ThreadPoolBuilder::new()
        .num_threads(threads.max(1) as usize)
        .thread_name(move |i| format!("hydrobem-{name}-{i}"))
        .build()
        .map_err(|e| Error::internal(format!("cannot start the {name} threads: {e}")))
}

/// The frequency workers of a solve, each with its inner pool.
pub struct FrequencyPools {
    workers: ThreadPool,
    inner: Vec<ThreadPool>,
}

impl FrequencyPools {
    /// frequency_threads workers (no more than n frequencies, at least one) of inner_threads each.
    pub fn new(frequency_threads: i32, inner_threads: i32, n: usize) -> Result<FrequencyPools> {
        let workers = (frequency_threads.max(1) as usize).min(n.max(1));
        let inner = (0..workers).map(|_| pool("inner", inner_threads)).collect::<Result<Vec<_>>>()?;
        Ok(FrequencyPools { workers: pool("frequency", workers as i32)?, inner })
    }

    /// f on the inner pool of the first worker (the limits, one after the other).
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.inner[0].install(f)
    }

    /// f(i, w) for i in 0..n, each on the inner pool of the worker w that takes it (the next free
    /// worker takes the next item). A cancel request of the solve is checked before every item;
    /// an error in an item stops the remaining ones and the first error recorded is
    /// returned; a panic in an item propagates once the workers have stopped. A worker with no
    /// item left is idle (`SolveContext::worker_idle`) until the others end.
    pub fn for_each(&self, n: usize, ctx: &SolveContext, f: impl Fn(usize, usize) -> Result<()> + Sync) -> Result<()> {
        let next = AtomicUsize::new(0);
        let err: Mutex<Option<Error>> = Mutex::new(None);
        self.workers.broadcast(|b| {
            let w = b.index();
            let inner = &self.inner[w];
            loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= n {
                    break;
                }
                if let Err(e) = ctx.check_cancel().and_then(|_| inner.install(|| f(i, w))) {
                    let mut g = err.lock().unwrap_or_else(|p| p.into_inner());
                    if g.is_none() {
                        *g = Some(e);
                    }
                    next.store(n, Ordering::SeqCst);
                    break;
                }
            }
            ctx.worker_idle(w);
        });
        match err.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The n items of a second-order step, as for_each: f(item) on the inner pool of the worker
    /// that takes it, the item begun on that worker and ended with the outcome of f; the step
    /// ended with the outcome of all.
    pub fn for_each_item(&self, n: usize, step: &StepWork, f: impl Fn(&ItemWork) -> Result<()> + Sync) -> Result<()> {
        let r = self.for_each(n, step.ctx, |i, w| {
            let it = step.item(i, Some(w));
            let r = f(&it);
            it.end(&r);
            r
        });
        step.end(r.is_ok());
        r
    }
}

/// The rows of a second-order step on the frequency workers, each row an item of the step: the
/// parts of a row in parallel on the inner pool of its worker, in their order (the same values on
/// any thread), the cancel request checked before every part. An error or a cancel stops the
/// rows; `finish` returns it.
pub struct StepRows<'a> {
    pools: FrequencyPools,
    step: StepWork<'a>,
    err: Mutex<Option<Error>>,
}

impl<'a> StepRows<'a> {
    /// The rows of `step` (n of them) on frequency_threads workers of inner_threads each.
    pub fn new(step: StepWork<'a>, frequency_threads: i32, inner_threads: i32, n: usize) -> Result<StepRows<'a>> {
        Ok(StepRows { pools: FrequencyPools::new(frequency_threads, inner_threads, n)?, step, err: Mutex::new(None) })
    }

    /// part(a, k) for every key k of keys(a), the rows a in 0..n: the values of each row in the
    /// order of its keys (a row not computed after an error or a cancel is empty).
    pub fn rows<K: Sync, T: Send>(
        &self,
        n: usize,
        keys: &(dyn Fn(usize) -> Vec<K> + Sync),
        part: &(dyn Fn(usize, &K) -> T + Sync),
    ) -> Vec<Vec<T>> {
        let out: Vec<Mutex<Vec<T>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();
        let r = self.pools.for_each_item(n, &self.step, |it| {
            let ks = keys(it.index());
            let done = AtomicUsize::new(0);
            let v = ks
                .par_iter()
                .map(|k| {
                    it.check_cancel()?;
                    let x = part(it.index(), k);
                    it.report(done.fetch_add(1, Ordering::Relaxed) + 1, ks.len());
                    Ok(x)
                })
                .collect::<Result<Vec<T>>>()?;
            *out[it.index()].lock().unwrap_or_else(|p| p.into_inner()) = v;
            Ok(())
        });
        if let Err(e) = r {
            *self.err.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
        }
        out.into_iter().map(|m| m.into_inner().unwrap_or_else(|p| p.into_inner())).collect()
    }

    /// The error that stopped the rows (ErrorKind::Cancelled for a cancel), if any.
    pub fn finish(self) -> Result<()> {
        match self.err.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(e) => Err(e),
            None => self.step.check_cancel(),
        }
    }
}

/// The first count rows of `width` elements of v, in parallel (rows of no elements when width is 0).
pub fn par_rows<T: Send>(v: &mut [T], width: usize, count: usize) -> impl IndexedParallelIterator<Item = &mut [T]> {
    if width == 0 {
        Either::Left((0..count).into_par_iter().map(|_| <&mut [T]>::default()))
    } else {
        Either::Right(v[..count * width].par_chunks_mut(width))
    }
}

/// The rows of par_rows, or None for every row when v is empty (an array not computed).
pub fn par_rows_opt<T: Send>(
    v: &mut [T],
    width: usize,
    count: usize,
) -> impl IndexedParallelIterator<Item = Option<&mut [T]>> {
    if v.is_empty() {
        Either::Left((0..count).into_par_iter().map(|_| None))
    } else {
        Either::Right(par_rows(v, width, count).map(Some))
    }
}

/// f on the items in groups of `group`, the items of a group in parallel: at most `group` items
/// (and their workspaces) are in progress at once, however the pool's threads steal work. The
/// first error stops the remaining groups.
pub fn try_for_each_grouped<T: Send>(
    items: &mut [T],
    group: usize,
    f: impl Fn(&mut T) -> Result<()> + Sync,
) -> Result<()> {
    for g in items.chunks_mut(group.max(1)) {
        g.par_iter_mut().try_for_each(&f)?;
    }
    Ok(())
}

/// f on the items in batches of `batch`, the items of a batch in parallel, then `write` with the
/// results of the batch in the order of its items (e.g. written into a matrix by write_blocks).
/// The first error stops the remaining batches.
pub fn in_batches<I: Sync, O: Send>(
    items: &[I],
    batch: usize,
    f: impl Fn(&I) -> Result<O> + Sync,
    mut write: impl FnMut(Vec<O>),
) -> Result<()> {
    for b in items.chunks(batch.max(1)) {
        write(b.par_iter().map(&f).collect::<Result<Vec<O>>>()?);
    }
    Ok(())
}

/// A block [r0, r0 + nr) x [c0, c0 + nc) of each of several column-major matrices, computed by one
/// item of a parallel loop and written into the matrices after (write_blocks).
pub struct Block<T> {
    pub r0: usize,
    pub nr: usize,
    pub c0: usize,
    pub nc: usize,
    v: Vec<T>, // [matrix][column][row]
}

impl<T: Copy> Block<T> {
    pub fn new(r0: usize, nr: usize, c0: usize, nc: usize, matrices: usize, zero: T) -> Block<T> {
        Block { r0, nr, c0, nc, v: vec![zero; matrices * nr * nc] }
    }
    /// Entry (r, c) of the matrix `mat` (rows and columns of the matrices).
    #[inline]
    pub fn set(&mut self, mat: usize, r: usize, c: usize, x: T) {
        let i = (mat * self.nc + c - self.c0) * self.nr + r - self.r0;
        self.v[i] = x;
    }
}

/// The blocks written into the column-major matrices `mats` of `rows` rows each, the column bands of
/// every matrix in parallel. The blocks must not overlap.
pub fn write_blocks<T: Copy + Send + Sync>(mats: &mut [&mut [T]], rows: usize, blocks: &[&Block<T>]) {
    const BAND: usize = 64; // columns
    if rows == 0 {
        return;
    }
    let mut bands: Vec<(usize, usize, &mut [T])> = Vec::new();
    for (mi, m) in mats.iter_mut().enumerate() {
        for (b, band) in m.chunks_mut(BAND * rows).enumerate() {
            bands.push((mi, b * BAND, band));
        }
    }
    bands.into_par_iter().for_each(|(mi, c0, band)| {
        let nc = band.len() / rows;
        for blk in blocks {
            for c in c0.max(blk.c0)..(c0 + nc).min(blk.c0 + blk.nc) {
                let src = &blk.v[(mi * blk.nc + c - blk.c0) * blk.nr..][..blk.nr];
                band[(c - c0) * rows + blk.r0..][..blk.nr].copy_from_slice(src);
            }
        }
    });
}

/// Single entries (index, value) of a vector computed by one item of a parallel loop, all in the
/// band `band` of the vector (write_entries).
pub struct Entries<T> {
    pub band: usize,
    pub e: Vec<(usize, T)>,
}

/// The entries written into v, its bands of band_len elements in parallel. No two entries may have
/// the same index.
pub fn write_entries<T: Copy + Send + Sync>(v: &mut [T], band_len: usize, lists: &[&Entries<T>]) {
    if band_len == 0 {
        return;
    }
    v.par_chunks_mut(band_len).enumerate().for_each(|(b, chunk)| {
        let o = b * band_len;
        for l in lists.iter().filter(|l| l.band == b) {
            for &(i, x) in &l.e {
                chunk[i - o] = x;
            }
        }
    });
}
