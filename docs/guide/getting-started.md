# Getting started

HydroBEM computes the linear and second-order wave loads on floating and fixed bodies: added mass
and radiation damping, exciting forces, motions (RAOs), pressures and free-surface elevations, mean
drift forces and quadratic transfer functions (QTFs). It is a panel method in the frequency domain:
the wetted surface of each body is described by a mesh of panels, and the flow is solved one wave
frequency at a time.

This page installs HydroBEM and walks through a first complete problem, a floating cylinder in
regular waves. The other pages of the guide each take up one topic and can be read in any order
once this one is done (§3).

## 1. Installation

HydroBEM is used from Python. The calculations are done by an engine written in Rust, which the
Python package `hydrobem` contains as a compiled module. There are two ways to install it.

**From a wheel.** A wheel is a ready-built package. Each release of the repository has one
attached for Windows on x86-64 and Python 3.11 or later; install it with
`pip install <the wheel file>`. A wheel has the programming interface of its own version (see
[CHANGELOG.md](../../CHANGELOG.md)); this guide describes version 0.3.0.

**From the source code.** Building from the source needs two tools besides Python:

- Rust 1.85 or later, installed with [rustup](https://rustup.rs) for the toolchain
  `x86_64-pc-windows-msvc`;
- the linker of the Visual Studio 2022 Build Tools, which rustup offers to install (the workload
  "Desktop development with C++").

Nothing else is needed: the engine is pure Rust, and the first build downloads the Rust libraries
it uses. Python 3.11 or later with NumPy runs HydroBEM; the tests and scripts of the repository
also use SciPy, mpmath, pytest and ruff, and the validation report Matplotlib. The file
`environment.yml` creates a conda environment with all of them. In a terminal with that Python
environment active, in the folder of the repository:

```bat
pip install .
```

This builds the engine and installs the package. To work on HydroBEM itself, build it in place
instead and run the fast tests (about a minute):

```bat
set HB_PYTHON_ENV=<Python environment directory>
scripts\build.bat Release
scripts\test.bat fast
```

`scripts\test.bat full` runs every test and takes several minutes. HydroBEM is built and tested on
Windows x86-64; the Linux scripts (`scripts/build.sh`, `scripts/test.sh`) are kept but not tested.

The engine is compiled for processors with the AVX2 and FMA instructions, which most x86-64
processors made since 2013 to 2015 have. The settings for this are in `.cargo/config.toml`; if you
set `RUSTFLAGS` yourself, it replaces them and must name both instructions too.

To check the installation, start Python and ask for the build information:

```python
import hydrobem as hb

print(hb.build_info())
```

### 1.1 Troubleshooting

Table 1 lists the messages most often seen during installation.

*Table 1. Messages during installation, their causes and remedies.*

| Message | Cause and remedy |
|---|---|
| `build.bat: cargo not found` | Rust is not installed, or the terminal was opened before it was. Install Rust with rustup and open a new terminal. |
| `linker link.exe not found` (from cargo) | The Visual Studio Build Tools are missing. Install them with the workload "Desktop development with C++". |
| `cannot import name '_core' from 'hydrobem'` | The compiled module is missing: it was not built, or the build could not replace it because a Python process still had the old one loaded. Close Python and run `pip install .` again (or `scripts\build.bat` for a checkout). |
| `this processor does not report AVX2 and FMA …` | The processor lacks the instructions the engine is compiled for. If the processor does have them and the system reports them wrongly, set the environment variable `HYDROBEM_SKIP_CPU_CHECK=1`. |
| `ImportError: DLL load failed` from NumPy (conda on Windows) | The folder `Library\bin` of the conda environment is not on `PATH`. Activate the environment, or run the tests through `scripts\test.bat`, which adds it. |
| `test.bat: set HB_PYTHON_ENV …` | The test script does not know which Python to use. Set `HB_PYTHON_ENV` to the folder of the environment (conda or venv), or activate the conda environment. |

## 2. A first problem: a floating cylinder

The problem is a vertical circular cylinder of radius 1 m and draft 0.5 m floating freely in deep
water. We want its added mass and damping, the exciting force of waves from two directions, and its
motions. Every problem in HydroBEM is set up in the same order: the mesh, the body, the waves, what
to compute, and then the solve.

### 2.1 The model

All inputs of a problem are held by one object, the *model*, `hb.Model()`. A new model is empty
apart from its defaults: deep water, gravity $g = 9.80665\ \mathrm{m/s^2}$ and water density
$\rho = 1000\ \mathrm{kg/m^3}$.

```python
import numpy as np
import hydrobem as hb

m = hb.Model()
```

### 2.2 The mesh

A *mesh* describes the wetted surface of a body by flat quadrilateral *panels* below the free
surface $z = 0$. The four corners of each panel are ordered so that its normal points out of the
body into the water. Usually a mesh comes from a meshing program as a GDF file, and one line reads
it:

```python
# m.meshes.load("cylinder", "cylinder.gdf")
```

So that this example runs without a file, we make the panels with a few lines of NumPy instead.
The cylinder is symmetric about the planes $x = 0$ and $y = 0$, so only one quarter of it is
meshed; the engine generates the other three quarters by reflection.

```python
def cylinder_panels(radius, draft, n_around=16, n_down=6, n_bottom=6):
    """One quarter (x >= 0, y >= 0) of the side and bottom of a vertical cylinder,
    the corners ordered so that the normals point into the water."""
    phi = np.linspace(0.0, np.pi / 2, n_around + 1)
    z = np.linspace(0.0, -draft, n_down + 1)
    r = np.linspace(radius, 0.0, n_bottom + 1)

    def p(radius, angle, depth):
        return [radius * np.cos(angle), radius * np.sin(angle), depth]

    side = [[p(radius, phi[j], z[i]), p(radius, phi[j], z[i + 1]),
             p(radius, phi[j + 1], z[i + 1]), p(radius, phi[j + 1], z[i])]
            for i in range(n_down) for j in range(n_around)]
    bottom = [[p(r[i], phi[j], -draft), p(r[i + 1], phi[j], -draft),
               p(r[i + 1], phi[j + 1], -draft), p(r[i], phi[j + 1], -draft)]
              for i in range(n_bottom) for j in range(n_around)]
    return np.array(side + bottom)


m.meshes.add("cylinder", kind="panels", vertices=cylinder_panels(1.0, 0.5),
             symmetry="xz_and_yz")
```

`kind="panels"` says that the mesh is made of flat panels, and `symmetry="xz_and_yz"` that the
stored quarter is to be reflected about both planes. Meshes, their kinds and the ways to read
them are the subject of [Meshes and bodies](meshes-and-bodies.md).

### 2.3 The body

A *body* is a rigid structure in the water. It names its mesh and carries its mass properties.
Here the mass is that of the displaced water, the centre of gravity lies 0.2 m below the
waterline, and the radii of gyration are 0.6 m in roll and pitch and 0.7 m in yaw:

```python
m.bodies.add("cylinder", mesh="cylinder", mass="displaced",
             center_of_gravity=[0.0, 0.0, -0.2], radii_of_gyration=[0.6, 0.6, 0.7])
```

### 2.4 The waves

The waves are part of the *environment*. We give three wave periods and two headings. A heading
is the direction the waves travel, in degrees counter-clockwise from the $+x$ axis, so 0 means
waves travelling along $+x$:

```python
m.environment.update(frequency_input="periods", periods=[2.0, 4.0, 8.0])   # s
m.environment.headings = [0.0, 45.0]                                         # degrees
```

The frequencies can be given as periods, as here, or as radian frequencies $\omega$ in rad/s
(`omegas`); `frequency_input` says which. [Water, waves and options](environment-and-options.md)
explains this and the other items of the environment.

### 2.5 Checking before the solve

Before solving, it is worth checking that the geometry is what you meant. The hydrostatics of the
body need neither a solve nor the waves. Like the other queries of a body, they check only the
rules that the body relies on: the values of the water (depth, gravity and density) and the
body's own geometry and surfaces. The waves, the options and the other bodies are checked by
`m.check()` and the solve:

```python
h = m.bodies["cylinder"].hydrostatics()
print(h.volume, h.metacentric_height)   # m^3, and GM in roll and pitch, m
```

The volume should be $\pi \cdot 1^2 \cdot 0.5 \approx 1.57\ \mathrm{m^3}$, slightly less for the
polygon of the mesh. `m.check()` checks the whole model against the rules of the engine and
the combinations it solves, and reports the first problem it finds
([Working with the model](working-with-the-model.md#4-checking-a-model)).

### 2.6 What to compute, and the solve

By default HydroBEM solves the radiation and diffraction problems, which give the added mass,
damping and exciting forces. The motions need one more switch:

```python
m.first_order.motion = True
results = m.solve()
print(results.summary())
```

`m.solve()` returns the *results*. `summary()` gives a short overview of what was computed.

### 2.7 Reading the results

The results are NumPy arrays in SI units, per unit wave amplitude. The added mass has one entry
per frequency, per force direction and per motion; the motions one per frequency, per heading and
per degree of freedom. The degrees of freedom of a body are numbered 0 to 5 in the order surge,
sway, heave, roll, pitch and yaw, so heave is index 2:

```python
print(2 * np.pi / results.omegas)          # the periods, s
print(results.added_mass[:, 2, 2])         # heave added mass, kg
print(results.radiation_damping[:, 2, 2])  # heave damping, kg/s
print(abs(results.motion[:, 0, 2]))        # heave RAO in the 0-degree waves, m/m
```

The same values can be read by body and by name, which is clearer when a model has several
bodies:

```python
print(results.coefficient("added_mass", ("cylinder", "heave"), ("cylinder", "heave")))
```

[Reading results](results.md) describes every result and its axes.

### 2.8 Saving the work

A model is saved as a *model file* (`.hbm`), which holds every input including the meshes. The
results are saved as a *results file* (`.hbr`), which also holds the model that was solved. The
results can also be written as numeric files in the WAMIT format for other programs:

```python
m.save("out/cylinder.hbm", overwrite=True)
results.save("out/cylinder.hbr", overwrite=True)
results.write_wamit("out/cylinder", overwrite=True)    # cylinder.1, .3, .4, .hst, .out, ...
```

Without `overwrite=True` an existing file is never replaced: the call raises `FileExistsError`.
A saved model is read back with `hb.Model.load("out/cylinder.hbm")`, and saved results with
`hb.load_results("out/cylinder.hbr")`.

The script `examples/cylinder/run.py` solves a version of this problem with every input of the
model written out by name with its default value; the other folders of `examples/` are complete
scripts of published cases.

## 3. Where to go next

The other pages of the guide each take one topic:

- [Working with the model](working-with-the-model.md): how the inputs are organized, how they are
  set, changed and checked, and the model and results files;
- [Meshes and bodies](meshes-and-bodies.md): reading meshes, choosing parts of a mesh, checking
  the geometry, mass properties, several bodies, trim;
- [Water, waves and options](environment-and-options.md): depth, walls and channels, frequencies
  and headings, what the first-order solve computes, the low-order and higher-order methods;
- [Reading results](results.md): the results, their axes and units, and the output files;
- [Special surfaces](special-surfaces.md): removing irregular frequencies, thin and porous plates,
  air cushions, plates on the free surface, damping lids and field points;
- [Tanks and generalized modes](tanks-and-generalized-modes.md): liquid in internal tanks, and
  deformations and other motions beyond the six rigid modes;
- [Mean drift and second-order loads](second-order-loads.md): mean drift forces, which method to
  trust, QTFs and Newman's approximation;
- [Performance, threads and progress](performance.md): solving faster, the memory a solve needs,
  and following a running solve.

The theory behind each result is in the [theory chapters](../theory/README.md), and every input
and result is listed in the [reference](../reference/inputs-model.md).
