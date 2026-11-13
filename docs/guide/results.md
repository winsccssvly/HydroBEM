# Reading results

A solve returns a results object that holds every quantity it computed as NumPy arrays. This page
explains how the arrays are laid out, how to pick out the coefficient of one body and one mode,
and how to write the results to files for other programs.

The examples continue [Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder):
the model `m` of the floating cylinder and its results `results`.

## 1. Quantities and axes

Each result is an array whose axes have fixed meanings, written below with four letters:

- **F**, the *ordinary frequencies*, in the order given (`results.omegas`, in rad/s); the limits
  $\omega = 0$ and $\omega = \infty$ are not on this axis (§2);
- **H**, the headings (`results.headings`, in degrees);
- **D**, the *coordinates*, the degrees of freedom of all bodies: six rigid coordinates per body
  (surge, sway, heave, roll, pitch, yaw, in this order), followed by the body's generalized and
  pressure coordinates, if it has any ([Tanks and generalized modes](tanks-and-generalized-modes.md));
  `results.coordinates` describes each one;
- **K**, the *radiation columns*, the coordinates whose radiation problem was solved: the free
  rigid modes of each body (its item `modes`) and its generalized and pressure coordinates.
  `results.radiation_mode_ids` gives the coordinate of each column, counted from 1.

The main first-order results are listed in Table 1. The values are in SI units per unit wave
amplitude: forces per metre of amplitude in N/m, moments in N m/m, and so on. Complex values are
amplitudes with the time factor $e^{+i\omega t}$, as defined in
[theory §1.1](../theory/01-boundary-value-problem.md#time-dependence).

*Table 1. The main first-order results and their axes.*

| Result | Axes | Meaning |
|---|---|---|
| `added_mass`, `radiation_damping` | F, D, K | the force on coordinate D from a unit motion of column K |
| `excitation.diffraction` | F, H, D | the exciting force by pressure integration |
| `excitation.froude_krylov`, `excitation.scattering` | F, H, D | its incident-wave and scattered parts |
| `excitation.haskind` | F, H, K | the exciting force by the Haskind relation |
| `motion` | F, H, D | the motions (RAOs), with `first_order.motion` |
| `hydrostatics.stiffness` | body, 6, 6 | the hydrostatic restoring of each body, about its rotation centre |

```python
print(results.added_mass.shape)          # (3, 6, 6): 3 frequencies, 6 coordinates, 6 columns
print(results.motion.shape)              # (3, 2, 6): 3 frequencies, 2 headings, 6 coordinates
print([c.name for c in results.coordinates])
```

A result that was not computed is `None`, never an array of zeros. `results.available` lists the
names of the results present, `hb.quantities()` describes every result the engine can compute,
with its axes and unit, and the whole tree of results is in
[reference/results §1](../reference/results.md#1-results-by-owner-and-by-name).

## 2. Results by body and by mode

With several bodies, counting coordinates by hand is error-prone. The results of one body are
found by its name, with its own coordinates and columns only:

```python
cyl = results.bodies["cylinder"]
print(cyl.added_mass.shape)              # F, the body's coordinates, the body's columns
print(cyl.excitation.diffraction[:, 0, 2])   # heave exciting force in the 0-degree waves
print(cyl.hydrostatics.volume)
```

One series is picked out by naming the body and the mode of the row (and of the column, for an
added mass or damping); `coordinate` gives the position of a mode on the D and K axes:

```python
a33 = results.coefficient("added_mass", ("cylinder", "heave"), ("cylinder", "heave"))  # [F]
x3 = results.coefficient("excitation_diffraction", ("cylinder", "heave"))              # [F, H]
d, k = results.coordinate("cylinder", "heave")       # 2, 2
```

What belongs to no single body stays on the results themselves: the axes, the added mass and
damping between bodies (the off-diagonal blocks of `results.added_mass`), the mean drift by
momentum conservation, and the field points in the sea.

The added mass at the frequency limits, when requested, is `added_mass_zero_frequency` and
`added_mass_infinite_frequency`, with the axes D and K.

### 2.1 Pressures, elevations and other outputs

The outputs at points are grouped with their points:

- `results.body_points`: pressure and velocity on the body (`first_order.body_pressure`,
  `body_velocity`), with the points, normals and areas of the panels where they are evaluated;
- `results.field_points`: elevation, pressure and velocity at the points in the water
  ([Special surfaces](special-surfaces.md#7-field-points));
- `results.lids[name]` and the pressure surfaces and tanks of each body: the elevations and
  pressures of those surfaces;
- `results.drift` and `results.qtf`: the second-order loads
  ([Mean drift and second-order loads](second-order-loads.md)).

The pressures are given as $p/(\rho g)$, in metres of water per metre of wave amplitude, and are
the dynamic pressure without its hydrostatic part. The radiation results are per unit motion of
each mode. The pressure or elevation on a body that moves is the diffraction part plus the
radiation parts weighted by the motions; HydroBEM gives the parts, and the sum is formed from
them:

```python
m.first_order.body_pressure = True
r = m.solve()
bp = r.body_points
k = list(r.radiation_mode_ids - 1)                 # the coordinate of each column
total = bp.pressure_diffraction + np.einsum("fnk,fhk->fnh", bp.pressure_radiation,
                                            r.motion[:, :, k])
print(total.shape)                                 # frequencies, panels, headings
```

## 3. Results and the model

The model keeps the results of its last solve as `m.results`, until an input changes
([Working with the model](working-with-the-model.md#5-the-model-and-its-results)). A results
object, once returned, does not change. It also holds:

- `results.model`, a copy of the model that was solved;
- `results.warnings`, the warnings of the model and of the solve, and `results.diagnostics`,
  every message of every severity;
- `results.summary()`, a short overview of what was computed.

The results hold what the model asked for. Other mass properties or other field points need a
new solve.

To animate the motions, `m.mode_displacements(points, bodies, elements)` gives the displacement of
every coordinate at points of the bodies, for example at the corners of `m.tessellation()`
([reference/queries §5](../reference/queries.md#5-hydrostatics-the-motion-equation-and-the-points)).

## 4. Files

The results are saved and read back with HydroBEM's own results file:

```python
results.save("out/cylinder.hbr", overwrite=True)
again = hb.load_results("out/cylinder.hbr")
```

For other programs, `write_wamit` writes the numeric output files of the WAMIT format and a
summary file `.out`. These files hold nondimensional values; `results.normalized(name)` gives a
result normalized in the same way:

```python
results.write_wamit("out/cylinder", overwrite=True)
print(results.normalized("added_mass")[:, 2, 2])
```

*Table 2. The numeric output files and the results they hold.*

| File | Contents | Results |
|---|---|---|
| `.1` | added mass and damping, with the frequency limits | `added_mass`, `radiation_damping`, `added_mass_zero_frequency`, `added_mass_infinite_frequency` |
| `.2`, `.2fk`, `.2sc` | exciting force by the Haskind relation | `excitation.haskind` |
| `.3`, `.3fk`, `.3sc` | exciting force by pressure integration, and its parts | `excitation.diffraction`, `.froude_krylov`, `.scattering` |
| `.4` | motions | `motion` |
| `.5p`, `.5vx`, `.5vy`, `.5vz` | pressure and velocity on the body | `body_points` |
| `.6p`, `.6vx`, `.6vy`, `.6vz` | elevation, pressure and velocity at field points, first those in the sea, then those in the tanks | `field_points`, and the tanks' |
| `.7`, `.8`, `.9` | mean drift by the control surface, by momentum conservation, by pressure integration | `drift.control_surface`, `.momentum_conservation`, `.pressure_integration` |
| `.hst` | hydrostatic restoring | `hydrostatics.stiffness`, `mode_restoring` |
| `.fpt` | the field points | `field_points.points` |
| `.pnl` | the panels of the body pressure | `body_points.points`, `.areas`, `.normals` |
| `.out` | a summary | |

The QTFs are kept in the results file only. The normalization and the layout of every file are in
[reference/file-formats](../reference/file-formats.md).

Results computed elsewhere can be written in the same formats: `hb.Results.of(m, added_mass=...,
...)` makes a results object of the model `m` from given arrays, which can then be saved or
written like any other.
