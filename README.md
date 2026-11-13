# HydroBEM

Frequency-domain panel-method (BEM) hydrodynamics: added mass, radiation
damping, exciting forces, motions, mean drift forces, second-order wave loads (QTFs), pressures,
velocities and free-surface elevations of floating and fixed bodies in waves. A Rust
engine (`hydrobem_core`) with a Python package (`hydrobem`, its extension module `hydrobem._core`
built with PyO3).

**Status:** 0.3.0, for Windows x86-64 (the processor needs AVX2 and FMA, as most
x86-64 processors since 2013-2015 have). The Linux build scripts are kept but not tested.

HydroBEM is developed independently from public literature
([docs/references.md](docs/references.md)) and checked against analytic
solutions and published results ([docs/validation/test-plan.md](docs/validation/test-plan.md);
the figures, the convergence studies and what is not compared with an independent solution:
[docs/validation/README.md](docs/validation/README.md), one page per case).
It reads and writes the WAMIT file formats: it reads GDF, SPL and CSF geometry files and the
panels of FDF free-surface files, writes GDF and SPL files, and writes the numeric output files.
Its own files are the
geometry file `.hbg` (one mesh of any kind but a trimmed one), the model file `.hbm` and the
results file `.hbr`.

## Features

- Low-order (flat panels, constant potential) and higher-order (flat or
  B-spline patches, B-spline potential) methods.
- Infinite and finite depth, the limits $\omega = 0$ and $\omega = \infty$; open sea, walls at
  $x = 0$ and/or $y = 0$, a channel (with a reflection factor).
- One or several bodies, placed and yawed; symmetry sectors.
- Removal of irregular frequencies (interior free surfaces), damping lids,
  thin (dipole) sheets, prescribed-pressure surfaces (air cushions), and free-surface plates
  (pistons, mats and covers, also on trimmed hulls and on the free surfaces of tanks).
- Radiation and diffraction (scattered or total), Haskind and pressure
  forces, motions (RAO), hydrostatics.
- Pressure and velocity at field points and on the body, free-surface
  elevation; mean drift by momentum conservation, by pressure integration and through a
  control surface, of one heading or of every pair of headings.
- Second-order loads (QTFs, difference and sum frequency, unidirectional or bidirectional): the
  quadratic part by pressure integration or through a control surface, the force of the
  second-order potential (direct and indirect, fixed or floating bodies, also with generalized
  modes, in the open sea or by one wall; low order), and Newman's approximation from any drift.
- Internal tanks, generalized modes (also in the pressure drift and the second-order loads),
  trimmed waterlines, dipole dampers, wavemakers in the walls.
- WAMIT-format numeric files (`.1 .2 .3 .4 .5 .6 .7 .8 .9 .hst …`) and an `.out`
  summary.
- A Python API (`hb.Model`), the Rust API of the engine and a C ABI (`hydrobem.dll`,
  `hydrobem.h`) for other programs; the progress of a running solve at any instant.

Not every feature works with every other: the supported combinations are in
the [input reference](docs/reference/inputs-rules.md#2-supported-combinations).

## Quick start

```python
import hydrobem as hb

m = hb.Model()
# a floating vertical cylinder (radius 1 m, draft 0.5 m) of the analytic geometry: its side and
# bottom, each 8 x 4 spans of order 3; or a mesh file: m.meshes.load("cylinder", "cylinder.gdf")
m.meshes.analytic("cylinder", "circular_cylinder", [(8, 4, 3, 3, 3, 4), (8, 4, 3, 3, 3, 4)],
                  radius=1.0, draft=0.5)
m.bodies.add("cylinder", mesh="cylinder", mass="displaced", center_of_gravity=[0, 0, -0.2],
             radii_of_gyration=[0.6, 0.6, 0.7])
m.options.higher_order = True            # the analytic geometry is solved by the higher order
m.environment.update(frequency_input="periods", periods=[2.0, 4.0, 8.0])  # s
m.environment.headings = [0.0]           # degrees
m.first_order.motion = True
results = m.solve()
print(results.added_mass[:, 2, 2], abs(results.motion[:, 0, 2]))   # heave A33, heave RAO
results.write_wamit("out/cylinder")
```

More in the [user guide](docs/guide/getting-started.md).

## Installation

The Quick start above is the API of 0.3.0. A wheel attached to a
[release](https://github.com/winsccssvly/HydroBEM/releases) (Windows x86-64, Python $\ge$ 3.11) has
the API of its version ([CHANGELOG.md](CHANGELOG.md)); or install from the source as below.

**From the source**, with Rust (rustup) and the Visual Studio 2022 Build Tools
(for the MSVC linker); nothing else needs to be installed:

```bat
git clone https://github.com/winsccssvly/HydroBEM.git
cd HydroBEM
pip install .
```

`pip install .` (and `pip install git+https://github.com/winsccssvly/HydroBEM.git`) builds the
extension module with cargo through [maturin](https://www.maturin.rs) and installs it with the
package; `maturin develop --release` installs it editable. One wheel serves every Python $\ge$ 3.11
(the stable ABI).
A wheel to distribute is built with `python scripts/build_wheel.py` (into `dist/`): as
`maturin build --release`, with the paths of the building machine removed from the module.

## Building

Requirements: Rust for `x86_64-pc-windows-msvc` ([rustup](https://rustup.rs); in this checkout it
uses the version of `rust-toolchain.toml`, the crates themselves need 1.85 or later) with the
Visual Studio 2022 Build Tools (for the linker), Python $\ge$ 3.11 with NumPy (SciPy, mpmath, pytest
and ruff for the tests; Matplotlib for the validation report). The engine is pure Rust: its
linear algebra is the faer crate, fetched by cargo. It is compiled for AVX2 and FMA
(`.cargo/config.toml`); a build without them is a compile error, and a `RUSTFLAGS` of one's own
must name them too.

```bat
rem Windows
set HB_PYTHON_ENV=<Python environment directory>
scripts\build.bat Release
scripts\test.bat fast
```

The build puts the extension module into the package in place (`python/hydrobem/_core.pyd`); the
tests import this checkout's package. Details: [docs/guide/getting-started.md](docs/guide/getting-started.md).

## Documentation

[docs/README.md](docs/README.md) lists everything: the input and output
reference, the theory and numerical methods, the development notes and the
test plan.

*Table 1. The folders of the repository and their contents.*

| Path | Contents |
|---|---|
| `crates/hydrobem/` | engine library (a Rust crate with a safe public API, [docs/reference/rust-api.md](docs/reference/rust-api.md); pure Rust, the linear algebra on faer); `src/tables/` generated function tables |
| `crates/hydrobem-python/` | the Python extension module `hydrobem._core` (PyO3) over the engine's Rust API |
| `crates/hydrobem-c/` | the C ABI `hydrobem.dll` with `include/hydrobem.h`, built from the sources ([docs/reference/c-api.md](docs/reference/c-api.md)) |
| `python/hydrobem/` | Python package (the model, its objects and the results, on `hydrobem._core`) |
| `examples/` | example scripts to read and run, each on its own (it makes its meshes itself and reads no file) and ending with a short check against a published or analytic value ([examples/README.md](examples/README.md); two need SciPy, `pip install hydrobem[examples]`): `cylinder/` writes out every input of a model by name with its default; `floating_hemisphere/`, `truncated_cylinder_finite_depth/`, `bottom_mounted_cylinder/`, `second_order_bottom_cylinder/` the cases of the [validation report](docs/validation/README.md); `two_truncated_cylinders/`, `two_submerged_spheres/`, `three_truncated_cylinders/`, `floating_owc_piston/` the multi-body cases of published studies; `floating_owc_chamber/` a floating oscillating water column with its air chamber and turbine; `floating_owc_cover/` the column covered by a light piston and `dock/` a floating dock of zero draft, both free-surface plates; `trimmed_moonpool_cover/` a pitched moonpool hull whose cover the engine generates from the trimmed waterline; `cylinder_with_heave_plate/` a truncated cylinder with a thick or a thin heave plate; `fish_tank_cylinders/` floating tanks of zero wall thickness with the sloshing inside; beside `run.py` (low-order panels) most folders have `run_higher_order.py`, the same case on exact-geometry patches ([docs/references.md](docs/references.md)) |
| `scripts/` | build and test scripts, generators of the tables and reference data |
| `tests/` | Python tests, reference data and regression baselines (the Rust tests are in `crates/*/tests/`) |
| `validation/` | the scripts of the validation report ([docs/validation/README.md](docs/validation/README.md)): its case pages, figures (the meshes among them) and numbers |
| `docs/` | documentation |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): how to report and contribute, the
sources that may be used, how methods and results are recorded, and the
tests. Security reports: [SECURITY.md](SECURITY.md).

## Citing

The citation of HydroBEM (its author, version and repository) is in
[CITATION.cff](CITATION.cff); the repository page offers it as APA and BibTeX.

## Licence

Copyright 2026 Sanghun Lee. Licensed under the
[Apache License, Version 2.0](LICENSE); see also [NOTICE](NOTICE).

The built extension module (and so every wheel) contains the Rust crates it is compiled with
(faer, rayon, num-complex, PyO3 and their dependencies) under their licences
([licenses/rust-crates.txt](licenses/rust-crates.txt)); the source code contains none of them.
Third-party software (these crates, NumPy, SciPy, mpmath, pytest, ruff, Matplotlib) and how
other parties' names and works appear: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

WAMIT is a trademark of WAMIT, Inc. HydroBEM reads and writes files in the WAMIT file formats; it
is not affiliated with, endorsed by or sponsored by WAMIT, Inc.
