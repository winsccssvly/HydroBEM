# Output files

## 1. Normalization (files only)

The files hold nondimensional values, the normalization of the WAMIT file formats **[B]**.
With $L$ the reference length (the mesh item `reference_length`) of the first body, $\rho$, $g$, unit
amplitude $A = 1$ m, and $r_i = 1$ for rotational coordinates, 0 for translations and pressure
coordinates:

- $\bar A_{ij} = A_{ij} / (\rho L^{3+r_i+r_j})$, $\bar B_{ij} = B_{ij} / (\omega \rho L^{3+r_i+r_j})$
- $\bar X_i = X_i / (\rho g A L^{2+r_i})$ (pressure, FK, scattering, Haskind)
- $\bar\xi_i = \xi_i L^{r_i} / A$
- radiation elevation per unit displacement: $\eta / L^{r_j}$; diffraction
  elevation: $\eta / A$
- $\bar C_{ij} = C_{ij} / (\rho g L^{2+r_i+r_j})$
- mean drift: $\bar F_i = F_i / (\rho g A^2 L^k)$, $k = 1$ for forces, 2 for moments
- field velocity: radiation $K L \nabla\phi_j / L^{r_j}$ ($K = \omega^2/g$), diffraction $V \omega L/(i g A)$

## 2. Numeric files

The files are in the WAMIT file formats **[B]**: their content, columns and fixed-width
layout.

Every computed quantity is written automatically (no file-type selector; Table 1).

*Table 1. The numeric files, their rows and when they are written.*

| File | Rows | Written when |
|---|---|---|
| `.1` | period, I, J, $\bar A$, $\bar B$ for every frequency, force row I and solved column J; the limits as rows with period $-1$ ($\omega = 0$) and 0 ($\omega = \infty$) and only $\bar A$ | radiation solved |
| `.2`, `.2fk`, `.2sc` | period, $\beta$, I, $\lvert \bar X \rvert$, phase (deg), Re, Im — Haskind force, its FK part, and Haskind $-$ FK | Haskind available |
| `.3`, `.3fk`, `.3sc` | same columns — pressure force, FK, scattering | diffraction solved |
| `.4` | same columns — RAO | `motion` |
| `.5p`, `.5vx`, `.5vy`, `.5vz`, `.pnl` | body pressure and velocity at the panel centroids, radiation and diffraction in separate blocks, each point labelled by two integers (M, K) in the record order of the format (one body: M the reflection, in the order identity, $-x$, $-x-y$, $-y$, and K the element; several bodies: $M = 1$, K running over the bodies, the reflections identity, $-x$, $-y$, $-x-y$, and the elements); `.pnl` centroid (global), area, normal (into the body) and $r \times n$ in body axes about the body origin | `body_pressure`, `body_velocity` |
| `.6vx`, `.6vy`, `.6vz` | the velocity components in the layout of `.6p` (radiation and total diffraction, normalized as §1) | `field_velocity` |
| `.6p` | radiation: period, point, (Re, Im) per solved column; diffraction: period, $\beta$, point, Re, Im | field points |
| `.fpt` | point, x, y, z in global metres | field points |
| `.hst` | I, J, $\bar C$ (hydrostatic with weight + pressure-surface restoring, all coordinates) | every body has a centre of gravity and a mass |
| `.8` | period, $\beta_1$, $\beta_2$, I (1, 2, 6), $\lvert \bar F \rvert$, phase, Re, Im — mean drift by momentum of the drift heading pairs: $\beta_1 = \beta_2$, or with `drift_bidirectional = true` every pair with $\beta_1$ not after $\beta_2$ in heading order | `second_order.drift_momentum_conservation` |
| `.7` | the columns of `.9` — mean drift by the momentum flux through the control surfaces | `second_order.drift_control_surface` |
| `.9` | period, $\beta_1$, $\beta_2$, I (1–6, the moment about the rotation centre R, then $-4$, $-5$, $-6$ for the moment about the moving R, of $\beta_1 = \beta_2$ only), $\lvert \bar F \rvert$, phase, Re, Im — mean drift by pressure integration of the drift heading pairs | `second_order.drift_pressure_integration` |

- For ordinary frequencies, the first column is period (s) with
  `frequency="period"` and $\omega$ (rad/s) with `"omega"`. The `.1` endpoint
  sentinel is always $-1$ for $\omega = 0$ and 0 for $\omega = \infty$, independent of the
  `frequency` setting.
- `.6p` row order is deterministic. For each ordinary frequency in input
  order, write the radiation block first: field points in input order, with
  solved radiation-column pairs in requested mode order on each row. Then
  write the diffraction block: headings in input order, and within each
  heading the field points in input order. An unavailable block is omitted.
  `.fpt` lists the same field points once, in input order. **[P]/[H]** The field points of the
  files are those of the sea (`field_points`), then those of every tank of the bodies in order
  (`tanks`, placed with their bodies, global): the radiation rows of a tank's points are its
  `tanks_*_radiation`, its diffraction rows zero (no wave inside a tank). A file has its
  diffraction block when there are points in the sea.
- Without radiation columns ($K = 0$: no free, generalized or pressure coordinate is
  solved) there is no radiation block in `.5*` and `.6*`, no `.1`, `.2`, `.2fk`, `.2sc`
  file and no radiation table in `.out`.
- Phase in degrees, 0 for a zero value; modulus/phase from the complex value.
- Layout **[B]**: ASCII, CRLF line
  ends including the last line; first line a header naming the producer
  (HydroBEM), the file name and the export time; numbers as Fortran
  `E14.6` (frequency/heading in 0P form, coefficients in 1P form), indices `I6`
  (`I10` for field points), `.hst` records `2I6, 1P, E15.6`; two-digit
  exponents (values that need three digits are an overflow error; non-finite
  values are an error).
- Distinct frequencies or headings that collide at the printed precision are
  an error.

## 3. The standalone `.out`

A self-contained summary in the layout of the WAMIT-format `.out` file, its header line
`HydroBEM <version>  summary` naming the producer, then the method (`Higher-order method` or
`Low-order panel method`), the
fluid parameters (gravity, length scale, depth, density), per body the
geometry label, panel or patch counts (and potential coefficients for higher
order), thin-sheet count, the body position and yaw,
interior-surface method, hydrostatics (volumes, buoyancy centre, centre of
gravity, mass, water-plane area, C33..C56 dimensional) when available, the
rigid inertia matrix, the list of numeric outputs, the frequency convention,
the radiation mode order, pressure-surface settings, force method,
logarithmic integration, the diffraction formulation solved (`"auto"` resolved), boundary, one
line per lid; the total restoring and
additional damping matrices when the motion matrices exist; then per period
the added-mass/damping table, Haskind and diffraction exciting forces, RAOs
(modulus and phase) and the field, lid and pressure-surface components, and
the added-mass tables of the limits. **[H]**

## 4. Writing rules

- The folder of the files is made when it is missing (as by every writer of
  the package: `m.save`, `m.save_results`, `Results.save`, `write_wamit`); no
  sidecars, NPZ or JSON files are created.
- The numeric files and the `.out` are written all or none: every file to a temporary sibling
  (`<stem>.hydrobem-*.tmp`), the existing targets moved to temporary names, the new files renamed
  to the targets and the old ones removed. A failure (a target that cannot be replaced, e.g. held
  open by a spreadsheet, is an I/O error) undoes the steps made: the old files are back under
  their names and no temporary file is left. The model and results files are written to a
  temporary sibling (`<path>.tmp*`) renamed when complete; temporaries are removed on failure. A
  mesh (GDF, SPL, `.hbg`) is written directly.
- Existing targets need `overwrite=True`; files of the same stem whose
  quantity is no longer computed are an error even with overwrite.
- Preflight before any write: the shape of every quantity on the axes of its model,
  availability, the `.4` rule (`motion`), frequency/heading collisions.
