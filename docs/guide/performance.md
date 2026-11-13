# Performance, threads and progress

The time of a solve grows with the square of the number of panels for the matrices and with its
cube for their solution, and it is repeated for every frequency. A large model therefore runs for
minutes or hours, and needs gigabytes of memory. This page shows how to use the cores of the
computer, how much memory a solve needs before it starts, and how to follow and stop a running
solve.

The examples continue [Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder)
with the model `m` of the floating cylinder.

## 1. Threads

Two options of `m.options` set the number of threads:

- `frequency_threads`: how many frequencies are solved at the same time;
- `inner_threads`: how many threads work on each of those frequencies, building its matrix and
  solving it.

The work that does not depend on the frequency uses all `frequency_threads` $\times$
`inner_threads` threads. The results are the same, to the last digit, for any number of threads.

```python
m.options.update(frequency_threads=2, inner_threads=2)
r = m.solve()
```

Each frequency thread holds the matrices of one frequency in memory. On a large mesh, fewer
frequency threads with more inner threads each therefore need less memory for the same number of
cores; on a small mesh with many frequencies, more frequency threads are faster. The choice is
explained in [theory §13.3](../theory/13-linear-algebra-and-threads.md#133-threads-determinism-and-memory).

## 2. Memory

`m.memory_estimate()` gives the memory a solve will need, in bytes, without solving:

```python
e = m.memory_estimate()
print(e.peak / 1e6, e.commit / 1e6)       # MB
```

`peak` is the memory the solve will actually use at its peak. `commit` is larger: it is the memory
the operating system must reserve, which includes buffers that every thread of the matrix
products sets aside but hardly touches. Before a solve the engine compares `peak` with the
physical memory available at that moment, and refuses a solve that needs more with a `ValueError`
that gives both numbers and the remedies: fewer frequency threads, fewer panels or, for the
second-order potential, a coarser free surface. Without this check the solve would fail on an
allocation, which ends the Python process. `m.options.memory_check = False` solves without the
check.

## 3. Symmetry

A mesh that stores one half or one quarter of a symmetric body (the item `symmetry`,
[Meshes and bodies](meshes-and-bodies.md#1-what-a-mesh-is)) makes the solve much cheaper: the
problem splits into independent parts, one for each way the flow can be symmetric or
antisymmetric about the planes, each with a fraction of the unknowns. A quarter mesh with two
planes of symmetry solves four smaller problems instead of one large one. Declare the symmetry of
the mesh whenever the bodies, their surfaces and the walls have it.

## 4. Following a solve

A function given to `m.solve` is called while the solve runs. A function of two arguments is
called after each frequency with the number done and the total:

```python
r = m.solve(lambda done, total: print(f"frequency {done} of {total}"))
```

A function of one argument receives an `hb.Progress` report at most every 0.1 s, and at every
finished frequency and every change of the phase of the solve. Its field `stage` says what the
reporting thread is doing: `"statics"` (the work that does not depend on the frequency),
`"assembly"`, `"lu"` and `"outputs"` (the three steps of a frequency), `"frequency_done"`,
`"phase"`, `"second_order"` and `"worker_idle"`:

```python
def report(p):
    if p.stage == "frequency_done":
        print(f"{p.frequencies_done} of {p.frequencies_total} after {p.seconds:.1f} s")


r = m.solve(report)
```

Every field of the report is described in `help(hb.Progress)`.

### 4.1 Stopping a solve

If the function returns `False`, the solve stops within a fraction of a second, also inside the
second-order steps. `m.solve` then raises `hb.SolveCancelledError` and the model is left without
results. `m.cancel()` does the same from another thread.

```python
try:
    m.solve(lambda done, total: done < 1)      # stop after the first frequency
except hb.SolveCancelledError:
    print("stopped")
```

## 5. The progress of a running solve

A graphical interface or a long script may want to show the progress of every frequency at once.
`m.progress()` returns the state of the running solve, or of the last one, at the moment it is
called. It can be called from any thread while `m.solve` runs on another; it never waits for the
solve and does not slow it down:

```python
import threading

done = threading.Event()
t = threading.Thread(target=lambda: (m.solve(), done.set()))
t.start()
while not done.wait(0.2):
    s = m.progress()
    print(s.phase, f"{s.done} of {s.total}", [round(f.fraction, 2) for f in s.frequencies])
t.join()
```

The answer is an `hb.SolveProgress` with the fields of Table 1.

*Table 1. The fields of a `SolveProgress`.*

| Field | Meaning |
|---|---|
| `running`, `seconds` | whether the solve is running, and the seconds since it began (until it ended) |
| `phase` | `"preparation"`, `"limits"`, `"first_order"`, `"second_order"`, `"finishing"`, then `"done"`, `"cancelled"` or `"failed"`; None before the first solve. The phase never goes back |
| `preparation` | the share done of the work that does not depend on the frequency, 0 to 1 |
| `limits` | one `hb.LimitProgress` per frequency limit ($\omega = 0$, $\omega = \infty$), in the order solved |
| `frequencies` | one `hb.FrequencyProgress` per ordinary frequency, in the order of the results |
| `done`, `total` | the frequencies and limits done, and their number |
| `workers` | one `hb.WorkerProgress` per frequency thread |
| `second_order` | one `hb.SecondOrderProgress` per step of the second order, in the order begun |

A `FrequencyProgress` has the `state` of the frequency (`"waiting"`, `"running"`, `"done"`,
`"failed"`, or `"cancelled"` when it was not solved because the solve was cancelled or another
frequency failed), the `stage` it is in (`"assembly"`, `"lu"` or `"outputs"`, and `"second_order"`
for an item of a second-order step that is not a frequency), the `fraction` of
its work done, from 0 to 1, and the `worker` that runs it. The fraction weighs the stages by their
cost as measured on the frequencies already done, never decreases, and is 1 once the frequency is
done. A display that asks a few times a second sees every running frequency move.

The frequencies are shared among the frequency threads, the *workers*, numbered from 0 to
`frequency_threads` $- 1$ for the whole solve. A worker takes the next frequency when it has
finished one; the frequency limits run on worker 0. A `WorkerProgress` has the `state` of the
worker (`"running"`, `"idle"` when no work is left for it in the current phase, `"done"`), the
`task` it is on (`"limit"`, `"frequency"` or the name of a second-order step), the `index` of that
item, and its `stage` and `fraction`.

The second-order loads are computed after the last frequency, on the same workers, as steps whose
items run like frequencies (Table 2). Each step is a `SecondOrderProgress` with its `step`,
`state`, `fraction` (the mean of its items') and `items`, each a `FrequencyProgress`.

*Table 2. The steps of the second order and their items.*

| Step | Items |
|---|---|
| `quadratic_pressure_integration`, `quadratic_control_surface` | the rows $\omega_a$ of the pairs of frequencies (each $\omega_a$ with every $\omega_b \ge \omega_a$) |
| `potential_first_order` | the frequencies of the first-order problem of the second-order potential, which adds the free-surface points to the model's frequencies |
| `potential_radiation` | the pair frequencies of its radiation problem |
| `potential_fits` | one item: the fits of the far field on the circles |
| `potential_outer` | the batches of radii of the outer integral |
| `potential_pairs` | the pair frequencies: the inner free surface and the body condition of their pairs |
| `potential_direct` | the pair frequencies of the direct method's solve |

The fits and the outer integral run on all threads at once; their items have no worker (`worker`
None), and every worker shows them as its task. With the second-order potential the memory
estimate of §2 includes its two sub-problems, which are solved one after the other.
