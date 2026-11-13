# Validation

Validation here means comparing HydroBEM's results with solutions that do not come from HydroBEM:
published tables and figures of the literature, closed-form analytic solutions, and semi-analytic
solutions (eigenfunction matching, multipoles, series) derived from published formulations and
implemented independently of the panel method in this repository. Where no such solution is known, a
case rests on the identities of the problem and on convergence, and its page says so.

Each case has a page of its own. It opens with the result in plain words, then describes the problem
and its meshes, the reference and its accuracy, the agreement, the convergence and any special topic
of the case. Both methods are compared wherever the engine has them: the low order (constant panels)
and the higher order (a B-spline potential on exact or spline geometry).

*Table 1. The cases: what each compares, against which reference, and the result.*

| Case | What is checked | Reference | Result | Page |
|---|---|---|---|---|
| Floating hemisphere | heave, surge $A$, $B$ over $0.1 \le Ka \le 10$; $A$ at $\omega = 0$, $\infty$; mesh convergence; irregular frequencies (both methods) | Hulme (1982) tables 1, 2, eqs. (3.18), (5.15) | Both methods agree with the tables, within 0.065 % (low order) and 0.037 % (higher order) of the largest tabulated value; the limits $\omega = 0$, $\infty$ within 0.029 % and 0.021 % | [page](cases/floating_hemisphere.md) |
| Truncated cylinder, finite depth | heave, surge $A$, $B$ at $h/a = 2$, 4; mesh convergence (both methods) | eigenfunction matching (Yeung 1981; Garrett 1971; Bhatta & Rahman 1995; Bhatta 2003) | Both methods agree with the reference, within 0.1 % (low order, 48 panels) and $1.7 \times 10^{-5}$ (higher order) | [page](cases/truncated_cylinder_finite_depth.md) |
| Bottom-mounted cylinder | surge exciting force; mean surge drift (pressure, momentum) (both methods) | MacCamy & Fuchs (1954); Havelock (1940); the analytic scattering series | Exciting force within 0.037 % and momentum drift within 0.19 % (low order, $n = 48$); $6.2 \times 10^{-6}$ and $3.4 \times 10^{-5}$ (higher order) | [page](cases/bottom_mounted_cylinder.md) |
| Dock (free-surface plate) | heave $A$, $B$ of a rigid disc in the free surface; $\omega = 0$ (both methods) | eigenfunction matching (Garrett 1971; Yeung 1981); Rayleigh (1896) | Eigenfunction matching within 0.086 % (low order) and $5.4 \times 10^{-5}$ (higher order); Rayleigh's piston within $5.1 \times 10^{-6}$ and $2.2 \times 10^{-5}$ | [page](cases/dock.md) |
| Two truncated cylinders | total $A$, $B$, $X$ of two interacting cylinders, finite depth (both methods) | Kagemoto & Yue (1986) fig. 4 (read), table 2; semi-analytic interaction | Semi-analytic solution within 0.54 % (low order) and $7.6 \times 10^{-5}$ (higher order); the figure read agrees to its reading | [page](cases/two_truncated_cylinders.md) |
| Three truncated cylinders | own and interaction $A$, $B$ (surge, pitch-surge), surge $X$ of three cylinders, finite depth (both methods) | Mavrakos (1991) figs. 5b, 5c (read); semi-analytic interaction | Semi-analytic solution within 0.42 % (low order) and 0.011 % (higher order); the figures read agree to about their reading | [page](cases/three_truncated_cylinders.md) |
| Two submerged spheres | $A$, $B$, $X$ of two interacting submerged spheres, deep water (both methods) | Wu (1995) figs. 2–10 (read); multipole solution | Multipole solution within 1.1 % (low order) and $2.2 \times 10^{-5}$ (higher order); the figures read agree to their reading | [page](cases/two_submerged_spheres.md) |
| Cylinder with a heave plate | $X$, $A$, $B$ in surge, heave, pitch; thick plate and thin plate (dipole sheet) (both methods) | Han et al. (2024) figs. 4, 6 (read) | The figures read agree within 2 % of the largest value read (higher order); $b_{33}$ near its small peak, read approximately, within 8 % | [page](cases/cylinder_with_heave_plate.md) |
| Fish tanks | added mass of the fluid in a circular tank forced in surge and pitch; surge and pitch RAOs of the tank free in waves (both methods) | Faltinsen & Timokha (2009) series, as Newman (2021); Newman (2021) fig. 2 (read) | Interior series within 1.6 % (low order) and 0.26 % (higher order) away from the sloshing resonance (low order: 18 % for the small $A_{15}$ of the non-resonant draft); the RAOs free in waves within 0.005 of the values read from Newman's Fig. 2 (higher order) | [page](cases/fish_tank_cylinders.md) |
| OWC chamber with a turbine | chamber pressure restrained and free in heave; largest power; capture width (both methods) | Mavrakos & Konispoliatis (2012) tables 1, 2, 5; Evans (1982) | The printed chamber pressures within 2.4 % (higher order) and 5.0 % (low order); the higher order meets the capture-width bound | [page](cases/floating_owc_chamber.md) |
| Second-order force, bottom-mounted cylinder | force of the second-order potential, sum and difference (low order only) | Kim & Yue (1990) table 3 | The indirect and the direct method agree with the published values within 1.1 % from 12 panels on (low order) | [page](cases/second_order_bottom_cylinder.md) |
| Truncated cylinder, second order | Newman's approximation; convergence of the potential QTF and the momentum drift in their parameters (low order only) | convergence only | Newman's approximation within 7 % of the full QTF at $\Delta\omega = 0.1$ rad/s; the free-surface parameters converged (no independent reference) | [page](cases/cylinder.md) |
| Floating OWC, two bodies | heave $A$, $B$ of a piston filling the column of a tube (both methods) | Raghavan et al. (2024) case 4 (geometry only); methods against each other | No independent reference; the fine low order converges to the higher order, within 0.42 % | [page](cases/floating_owc_piston.md) |
| Floating OWC cover (free-surface plate) | heave, pitch $A$, $B$, $X$ of a tube with a rigid cover over its moonpool (both methods) | identities (energy, Haskind); low against higher order | No independent reference; the low order converges to the higher order (4.8 % at 32 panels around); energy identity within $2.1 \times 10^{-5}$ | [page](cases/floating_owc_cover.md) |
| Green functions | infinite- and finite-depth free-surface Green functions and derivatives (the kernel alone) | integral representations at 30 digits (mpmath) | Within the target $2 \times 10^{-6}$ everywhere: largest error $4.7 \times 10^{-7}$ (infinite depth) and $3.0 \times 10^{-7}$ (finite depth) | [page](cases/green_functions.md) |
| Fixed constants of the method | sensitivity of every constant without an input (both methods) | variation about the value; arguments | Each of the 12 constants lies in a flat range of its sensitivity; the default higher-order discretization reaches its best agreement with Hulme's tables at 36 unknowns per quadrant | [page](cases/constants.md) |

Unless a page says otherwise, a result is the largest difference over the compared range relative to
the largest reference value of each quantity. Values read from published figures are compared to the
accuracy of the reading, stated on each page.

How the comparisons are normalized and which references are used is described in
[method](method.md); what is not measured against an independent solution is listed in
[not verified](not_verified.md); the tests and their thresholds are in the
[test plan](test-plan.md).

The examples of `examples/` show how to use HydroBEM and are a subset of these cases. The tests of
`tests/` are a fast subset with thresholds, run on every change; the pages hold the full comparison.
Every figure, number and page here is regenerated by the scripts of `validation/`
(`python validation/make_report.py`; `--quick` for a coarse check of the scripts, `--pages` to
rewrite the pages from `figures/numbers.json`).

The whole report takes about 27 minutes on 14 threads (each group's time in `figures/numbers.json`,
`run`).
