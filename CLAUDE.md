# HydroBEM development rules

*This is the public copy of the project instructions; the private development rules of the
project are kept in a separate repository of their own.*

HydroBEM is an independently developed frequency-domain panel-method (BEM) engine with a Python
package and a C ABI, written from public literature, publicly documented formats and analytic
solutions. Keep every file to the standard of a published open-source project: ours to
distribute, sources cited, nothing third-party committed.

## Start of a session

1. Read this file, then `docs/development/architecture.md` (the module map, the solve flow and
   the conventions) and `docs/development/rules.md`.
2. Read `docs/README.md` and the sections of `docs/` that the work touches.
3. Build and test before changing anything (a read-only review needs neither; never run two full
   test runs at once):
   - `scripts\build.bat Release` — the cargo build of the workspace: the engine, the extension
     module (copied to `python\hydrobem\_core.pyd`) and the C ABI `target\release\hydrobem.dll`,
     and the Rust tests.
   - `scripts\test.bat fast` while iterating (the tests not marked slow, about a minute) and
     `scripts\test.bat full` before a commit (every test, the regression baselines, the
     thread-count identity and the checks of the generated files; about 4 minutes, plus the build
     of the Rust test binaries after a change).
   - Set `HB_PYTHON_ENV` to the Python environment (or activate it); by hand, `PYTHONPATH` must
     name `python\`.
   - The Rust tests alone: `cargo test --release` (the component tests in
     `crates/hydrobem/tests/`, the C ABI tests in `crates/hydrobem-c/tests/`; `HB_TIMING=1` adds
     the kernel timings).
   - Lints: `cargo clippy --workspace --all-targets --release -- -D warnings`,
     `cargo fmt --all -- --check` (`cargo fmt --all` formats; `rustfmt.toml`) and `ruff check`.

## The repository

| Path | Contents |
|---|---|
| `crates/hydrobem/` | the engine (Rust) and its component tests |
| `crates/hydrobem-python/` | the Python extension module |
| `crates/hydrobem-c/` | the C ABI, its header and its tests |
| `python/hydrobem/` | the Python package (the model objects, the queries, the results) |
| `tests/python/` | the tests by feature, their helpers and the regression baselines |
| `validation/` | the scripts of the validation report |
| `scripts/` | the build and test scripts, the generators of the tables and of the generated parts of the documents |
| `examples/` | the worked examples, for the users to read |
| `docs/` | the documentation: `guide/`, `theory/`, `reference/`, `validation/`, `development/`, `references.md` |
| `licenses/` | the licence texts of the dependencies |

## Sources

HydroBEM is developed from:

- public literature (papers, books, theses, published tables and figures), cited in
  `docs/references.md`;
- publicly documented file formats, for the files it reads and writes;
- analytic solutions and our own high-precision reference computations (arbitrary precision,
  verified quadrature);
- standard numerical libraries (faer, NumPy, SciPy, mpmath).

Source code, tables and data of other programs are not copied, and no program is disassembled,
decompiled or inspected internally.

## Repository hygiene

- Never commit copies of papers, their conversions or excerpts of manuals beyond short citations.
  Published numbers used as test references are extracted by a script from a local copy and
  cited.
- Every table or constant in the engine comes from a script in this repository with a cited
  derivation; regenerate, never edit tables by hand (`scripts\test.bat` checks the generated
  files that are cheap to rebuild).
- No machine-specific paths or personal data in tracked files, nor in the commit messages; use
  environment variables.
- Third-party software and notices: `THIRD_PARTY_NOTICES.md`. Licence Apache-2.0, copyright
  Sanghun Lee (`LICENSE`, `NOTICE`), which cover every file: the source files carry no licence
  header. The scripts of `examples/` cite their sources in their docstring.
- Line endings: LF (`.gitattributes`; the Windows command scripts CRLF). A script that writes
  files writes LF: in Python `write_bytes`, or `write_text(..., newline="\n")` (plain
  `write_text` writes CRLF on Windows).
- After a change of the Rust dependencies (`Cargo.lock`) rewrite `licenses/rust-crates.txt` with
  `scripts/gen_crate_licenses.py` (`scripts\test.bat` checks it).

## Tests

- Component tests of the engine in `crates/hydrobem/tests/` (`core.rs` and its modules, the
  public API in `model_api.rs`, the thread pools in `pools.rs`, a test binary of its own).
- The C ABI in `crates/hydrobem-c/tests/` (`c_api.rs` through its functions, `c_api.c` compiled
  by `tests/python/test_c_abi.py`).
- The tests of the package by feature in `tests/python/test_<feature>.py` with `conftest.py`;
  the shared helpers in `meshes.py`, `bodies.py`, `problems.py`, `regression_problems.py` (test
  files never import each other).
- Mark a test of more than a few seconds `@pytest.mark.slow`.
- After an intended change of results, regenerate the regression baselines
  (`scripts/gen_regression_baselines.py`) and say in the commit message why they changed.
- New work is complete when its own tests pass (`docs/validation/test-plan.md`, plans A and B).

## The documents

- The documentation is divided by reader (`docs/README.md`): the user guide (`docs/guide/`), the
  theory in numbered chapters (`docs/theory/`), what to look up (`docs/reference/`), the test
  plan and the validation report (`docs/validation/`), and the rules, the design and the module
  map (`docs/development/`).
- Every method choice is described in the theory chapters with its parameters, derivation and
  accuracy, written as the present state; the code cites the section (`theory §N.M`). The
  citation forms and the notation of the formulas, figures and tables are in
  `docs/development/rules.md`; `scripts/check_doc_math.py` and `scripts/rewrite_doc_refs.py
  --check` check them (both run by `scripts\test.bat`).
- Keep the documents current with the code at the end of every work session: a new or removed
  module of `crates/*/src` or `python/hydrobem` changes the module map in
  `docs/development/architecture.md`; a change of the API version (`build_info.rs`) or of the ABI
  version (`HB_ABI_VERSION`) is written in `CHANGELOG.md`, which records the user-visible
  changes.

## Environment

Windows is the only supported target: a Rust toolchain (rustup, `x86_64-pc-windows-msvc`, the
version pinned in `rust-toolchain.toml`, the same in CI: a new Rust release is taken up by changing
it, then `cargo fmt --all` and clippy; the MSVC linker of the Visual Studio 2022 Build Tools) and a Python environment with NumPy, SciPy,
mpmath, pytest and ruff (`environment.yml`). The engine is pure Rust, its linear algebra on
faer. The Linux scripts are kept but not tested.
