# Third-party notices

HydroBEM's source code does not include third-party source code. This file
lists the software it is built with or uses, and how other parties' names and
works appear in the repository. The licence texts that a binary distribution
must carry are in `licenses/`.

## Compiled into the engine library and the extension module

The engine and its Python extension module are pure Rust: they are compiled from Rust crates,
fetched by cargo (`Cargo.lock`), not committed, and link no other library.

| Component | Use | Licence |
|---|---|---|
| faer 0.24 (Rust crate) and its dependencies (faer-traits, gemm, nano-gemm, private-gemm-x86, pulp, dyn-stack, spindle, qd, equator, generativity, reborrow, bytemuck, libm, raw-cpuid, atomic-wait, half, num_cpus) | the dense linear algebra of the engine (`linalg.rs`): the LU of each panel, the triangular solves and the matrix products of the blocked complex LU, the real LU and its solve | MIT, or MIT or Apache-2.0 (bytemuck also Zlib); zerocopy (a dependency of half) BSD-2-Clause or Apache-2.0 or MIT; atomic-wait BSD-2-Clause (with windows-sys and windows_x86_64_msvc 0.42, MIT or Apache-2.0); libm contains code derived from musl libc and FreeBSD (MIT and permissive notices, carried in `licenses/rust-crates.txt`); rustc-hash MIT or Apache-2.0; faer contains code derived from Eigen (MPL-2.0) and from LAPACK and SuiteSparse AMD/COLAMD (BSD-3-Clause), whose notices it carries |
| rayon 1.12 (Rust crate) and its dependencies (rayon-core, crossbeam, crossbeam-channel, crossbeam-deque, crossbeam-epoch, crossbeam-queue, crossbeam-utils, either) | the thread pools of a solve | MIT or Apache-2.0; crossbeam-channel contains code derived from Go (BSD-3-Clause) and from matching.go (CC-BY-3.0), whose licences it carries (`licenses/rust-crates.txt`) |
| num-complex 0.4 (Rust crate) and num-traits, num-integer | the complex types `Complex32`, `Complex64` of the engine | MIT or Apache-2.0 |
| PyO3 0.29, numpy 0.29 (Rust crates) and their dependencies (ndarray, matrixmultiply, libc, once_cell, ...) | the Python extension module (`crates/hydrobem-python`) | MIT or Apache-2.0; numpy BSD-2-Clause |
| Rust standard library | the standard library of the engine, compiled into it | MIT or Apache-2.0 |

The C ABI `hydrobem.dll` (`crates/hydrobem-c`, built from the sources) links the same engine
crates without PyO3 and numpy: a program that distributes it carries `LICENSE`, `NOTICE`, this
file and `licenses/rust-crates.txt` with it.

`licenses/rust-crates.txt` lists every crate compiled into the extension module with its version,
licence, authors and the licence and notice files of its package (for faer also the notices of the
code it derives from: LAPACK, Eigen, SuiteSparse); `scripts/gen_crate_licenses.py` writes it from
`Cargo.lock`. The Python wheel carries it, `LICENSE`, `NOTICE` and this file through
`license-files` in `pyproject.toml` (installed under `hydrobem-*.dist-info/licenses/`).

The source code of every crate, including the files of faer under the Mozilla Public License 2.0
(the code derived from Eigen), is available in the form compiled into the binaries from crates.io
(`https://crates.io/crates/<name>/<version>`, the versions of `Cargo.lock` and
`licenses/rust-crates.txt`) and from the repository named for each crate in
`licenses/rust-crates.txt` (faer 0.24.4: `https://codeberg.org/sarah-quinones/faer`).

## Runtime and development dependencies (Python)

| Package | Use | Licence |
|---|---|---|
| NumPy | arrays in the Python package and scripts | BSD-3-Clause |
| SciPy | scripts (special functions) and tests | BSD-3-Clause |
| mpmath | arbitrary-precision reference values and table generation (scripts only) | BSD-3-Clause |
| pytest | tests | MIT |
| ruff | lint | MIT |
| Matplotlib | the figures of the validation report (`validation/`, development only; not distributed with the package) | Matplotlib License (PSF-based, BSD-compatible) |

## Build tools

The Rust toolchain (rustc, cargo; MIT or Apache-2.0) and maturin (MIT or Apache-2.0) build the
engine and the wheel, with the linker of the Microsoft Visual Studio Build Tools on Windows. The
proc-macro and build-script crates of `Cargo.lock` run only in the build: none of their code is in
the binary or distributed with the project. The C runtime is linked dynamically
(`.cargo/config.toml`): the extension module uses the Visual C++ runtime (`vcruntime140.dll`) that
Python for Windows installs and the Universal C Runtime of Windows; the binary does not contain
the runtime (only the start-up code the linker adds to every DLL), and the project distributes
no runtime file.

## External programs and publications

- HydroBEM reads and writes some of the WAMIT file formats.
- Published papers and books are cited in `docs/references.md`.
  Copies of them are not included.

## Third-party data

Numerical values from published tables serve as test references. They are extracted by a script
from a local copy of the publication, cited in the file, and remain the data of their authors:
they are not covered by the Apache License of this project (`NOTICE`).

| File | Source | Extracted by |
|---|---|---|
| `tests/data/hulme1982_full.txt` | A. Hulme (1982), *J. Fluid Mech.* 121:443–463, tables 1 and 2 | `scripts/extract_hulme.py` |
| `tests/data/kim_yue1990.txt` | M.-H. Kim, D. K. P. Yue (1990), *J. Fluid Mech.* 211:557–593, table 3 | `scripts/extract_kim_yue.py` |

The other files of `tests/data/` are HydroBEM's own computations (`scripts/gen_*.py`).
