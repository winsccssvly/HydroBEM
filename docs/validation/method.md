# How the comparisons are made

This page states the conventions that every case page of the [validation report](README.md) follows:
which kinds of reference are used and how far each can be trusted, how a difference is measured, how
convergence is judged, how the meshes and figures are drawn, and which claims of the theory documents
and the test plan the cases check.

## Which references are used

Each case is compared with the most independent reference available. In order of preference these
are:

- **Analytic solutions** in closed form: Rayleigh's piston (the disc in a rigid plane at
  $\omega = 0$), the exciting force of MacCamy & Fuchs (1954), the capture width $1/k$ of an
  axisymmetric chamber (Evans 1982) and the energy of the radiated waves of an axisymmetric body
  (Newman 1977). They are exact.
- **Published tables**: Hulme (1982), Kim & Yue (1990), Kagemoto & Yue (1986, table 2) and Mavrakos &
  Konispoliatis (2012). The numbers are extracted from a local copy of the publication by a script of
  `scripts/` (`extract_hulme.py`, `extract_kim_yue.py`), or written into the case's script with the
  page and table cited. Their printed digits bound the agreement: four digits are a rounding of up to
  5e-5 of the value.
- **Semi-analytic solutions derived and implemented in this repository** from published formulations:
  eigenfunction matching of truncated cylinders (`scripts/gen_cylinder_reference.py`), the multiple
  scattering of groups of cylinders (`gen_cylinder_array_reference.py`), the multipoles of submerged
  spheres (`gen_sphere_array_reference.py`), the analytic scattering series of the bottom-mounted
  cylinder (`validation/bottom_cylinder.py`) and the interior series of a circular tank
  (`tests/python/problems.py`). Each converges in its truncation to well below the differences
  compared, and its page states how far. Each is independent of the panel method.
- **Values read from published figures**: where a publication gives its results as figures only,
  values are **read by eye** from the page images at a few abscissae each and written into the case's
  script. The page states the reading accuracy, typically 1 to 3 % of the values and more where a
  curve is steep. A difference within it is agreement, and a figure is never the only check of a
  quantity that a computed reference also covers. The examples of `examples/` carry copies of some
  of these values and of the published tables, so that each stays self-contained; the scripts of
  `validation/` and `tests/data` are the source, and `tests/python/test_example_references.py` checks
  that every copy agrees with it.
- **Identities and convergence** where no independent solution is known
  ([not verified](not_verified.md)): the two methods against each other as both refine, and the
  identities of the problem (energy, reciprocity, the Haskind relation, which gives the exciting
  force from the radiation potentials alone).

The reference values are evaluated at the frequencies of the solve, and the abscissae of the values
read are among the frequencies solved, so no interpolation enters a difference.

## How a difference is measured

Unless a page says otherwise, a difference is **the largest
$\lvert \text{HydroBEM} - \text{reference} \rvert$ over the compared range, relative to the largest
$\lvert \text{reference} \rvert$ of the same quantity over that range**. A quantity that passes
through zero is therefore not measured relative to its small values.

Values read from a figure are measured relative to the largest value read of the quantity. The
tables of published values give the signed relative difference per entry.

Units are SI with $\rho = 1000\ \mathrm{kg/m^3}$, $g = 9.80665\ \mathrm{m/s^2}$ and the wave amplitude
$A = 1$ m, except where a page states the reference's own units: the fish tanks are nondimensional
($\rho = g = 1$), and the OWC chamber uses $\rho = 1025\ \mathrm{kg/m^3}$ and $g = 9.81\ \mathrm{m/s^2}$.

## How convergence is judged

Agreement on one mesh could be luck, so the main cases are also solved on a sequence of refined
meshes. From three meshes $h_0 > h_1 > h_2$ the order $p$ is observed per value, assuming
$v_i = v^* + c\, h_i^p$. Richardson's (1911) extrapolation $v^*$ with that order is the limit of zero
panel size that this rate implies (`validation/common.py`, `observed_order`). An order near the one
expected of the method, and an extrapolation that lands on the reference, show that the remaining
difference is discretization.

## Methods and meshes

Each case is solved by the low-order method (constant panels) and, where the engine has it, by the
higher-order method (a B-spline potential on exact analytic, rational or spline geometry). The
potential QTF and some second-order outputs exist in the low order only
([theory §12.10](../theory/12-second-order-loads.md#1210-the-sub-problems)).

The meshes drawn on the pages are those of the comparison, taken from the engine's own queries
(`validation/mesh_figure.py`: `Model.panels` and `Model.tessellation`, joined by their keys). They are drawn with their reflections over the
symmetry planes and coloured by role: the wetted surface, the interior free surface (which removes
the irregular frequencies), damping lids, free-surface plates, pressure surfaces, thin sheets
(dipoles) and tank walls. A higher-order patch is drawn as a smooth tessellation with its sides
(dark) and the edges of the spans of its potential (grey). Closed bodies are cut open where their
inside matters.

## Claims checked

The statements of the theory documents and the test plan that the cases measure are listed in
Table 1; the numbers are on the case pages.

*Table 1. The claims checked: the claim and the document that makes it, the case that measures it
and the result.*

| Claim (document) | Case | Result |
|---|---|---|
| A4: hemisphere $\le$ 1e-3 at the finest of three meshes, converging (test-plan) | [floating hemisphere](cases/floating_hemisphere.md) | holds, $O(h^2)$ |
| A4 whole range with an interior surface, 48 graded divisions, $\le$ 1e-3 (test-plan, validation) | [floating hemisphere](cases/floating_hemisphere.md) | holds |
| the rational hemisphere against Hulme within 3e-4 (test-plan B) | [floating hemisphere](cases/floating_hemisphere.md) | holds at $Ka = 0.5$, 1, 2 |
| A5: $\le$ 2e-3 at 48, monotone, $\le$ 1e-3 extrapolated, order about 1.8 (test-plan, validation) | [truncated cylinder](cases/truncated_cylinder_finite_depth.md) | holds |
| A2: the Green functions within 2e-6 (theory §8) | [Green functions](cases/green_functions.md) | holds |
| Kim & Yue within 1.5 % (test-plan B) | [second-order force](cases/second_order_bottom_cylinder.md) | holds |
| the Kochin angles: $N \ge 2kR$ + about $10 (kR)^{1/3}$ for 1e-6 (theory §11.4) | [cylinder](cases/cylinder.md) | holds |
| panels and cells within 1e-2 (theory §12.8) | [cylinder](cases/cylinder.md) | holds at 0.2 m and finer |
| $R$: the coefficients do not depend on $R$ once $R$ is outside the near field (theory §12.12) | [cylinder](cases/cylinder.md) | within 1 % of the largest coefficient from 3 to 12 m |
| several bodies: the interaction of the multiple-scattering references (test-plan, `test_multiple_bodies.py`) | [two cylinders](cases/two_truncated_cylinders.md), [three cylinders](cases/three_truncated_cylinders.md), [two spheres](cases/two_submerged_spheres.md) | holds over the whole range |
| a free-surface plate: Rayleigh's piston and the dock (theory §9.7) | [dock](cases/dock.md) | holds, converging |
| the capture width of an axisymmetric pressure chamber (Evans 1982; `test_pressure_surfaces.py`) | [OWC chamber](cases/floating_owc_chamber.md) | holds in the higher order up to $ka = 2.5$ |
| a tank's added mass against the interior series (`test_internal_tanks.py`) | [fish tanks](cases/fish_tank_cylinders.md) | holds away from the sloshing resonance |
