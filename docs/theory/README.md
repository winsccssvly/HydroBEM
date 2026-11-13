# Theory

These chapters describe the mathematical problem HydroBEM solves and define everything it
computes. The problem is linear potential flow of an ideal, incompressible fluid, with
small-amplitude waves and motions, in the frequency domain. The chapters then explain how the
boundary-value problems are discretized, the parameters of every numerical component, why they
were chosen, and the accuracy each one is built and tested to.

They are written for a reader who knows the hydrodynamics of floating structures and wants to know
exactly what HydroBEM computes and how far its numbers can be trusted. The inputs that select the
methods are described in the [reference](../README.md) (`reference/inputs-*`).

## Sources and tags

The theory is the standard linear wave–body theory (Wehausen & Laitone 1960; Newman 1977). Results
taken from the literature are tagged **[P]** ([references.md](../references.md)); the other results
are derived here from the same equations. Two further tags mark what is HydroBEM's own (the full
list of [the tags](../README.md)):

- **[C]** a numerical choice of HydroBEM;
- **[B]** a convention of the file formats it writes.

The tests are listed in [validation/test-plan.md](../validation/test-plan.md). The error maps
written by the generating scripts are in [development/error-maps/](../development/error-maps/).

## Common rules

A few rules hold for every numerical component:

- An accuracy target is fixed first, then the method. Among the methods that meet the target
  everywhere, the cheapest is used. Every table and constant is produced by a script in `scripts/`
  from a high-precision reference and checked against it (an error map).
- The panel radius $a$ is the largest distance from the centroid to a vertex. The panel size of the
  higher order is the largest centre–corner distance of a potential panel.
- Gauss–Legendre quadrature with $n$ points, on an integrand analytic in a Bernstein ellipse of
  parameter $\rho$, errs by $O(\rho^{-2n})$. The switches between near and far rules in the
  chapters follow from this bound and from measured error maps.
- The **Grounds** of a section give the reasons of its choices: the alternatives tried, the
  measurements that decided between them, and the cost.

## Chapters

Each chapter gives the physics of its topic, then the method, then its accuracy. Its sections are
numbered within it (`§11.2`) and cited so from the code (`theory §11.2`) and from the other
documents. The chapters fall into groups: the boundary-value problem (1) and the quantities
computed from it (2 to 4); the methods (5 to 8, 13 and 14); the special surfaces and modes (9 and
10); the mean drift (11) and the second-order loads (12) (Table 1).

*Table 1. The chapters of the theory.*

| Chapter | Contents |
|---|---|
| [1. Coordinates and the boundary-value problem](01-boundary-value-problem.md) | Coordinates and conventions, the field equations and boundary conditions, the limits $\omega = 0$ and $\omega = \infty$ |
| [2. Radiation, diffraction and coefficients](02-radiation-and-diffraction.md) | Radiation, excitation, added mass and damping |
| [3. Motions and hydrostatics](03-motions-and-hydrostatics.md) | Rigid-body motions, hydrostatics, the integration of the motion matrices |
| [4. Field outputs](04-field-outputs.md) | Free-surface elevation, pressure and velocity at points: definitions and methods |
| [5. Irregular frequencies and interior surfaces](05-irregular-frequencies.md) | Irregular frequencies and the extended boundary-integral equation |
| [6. The low-order method](06-low-order-method.md) | Constant panels: Rankine integrals, the logarithmic singularity, symmetry sectors and walls |
| [7. The higher-order method](07-higher-order-method.md) | B-spline potentials on patches: static integrals, the wave part, discretization |
| [8. The free-surface Green function](08-green-function.md) | Infinite and finite depth, four points at a time, the frequency limits, walls and channel |
| [9. Special surfaces](09-special-surfaces.md) | Damping lids, prescribed-pressure surfaces, thin sheets and dampers, free-surface plates |
| [10. Generalized modes, internal tanks and wavemakers](10-generalized-modes-and-tanks.md) | Generalized modes, internal tanks, wavemakers in the walls |
| [11. Mean drift](11-mean-drift.md) | Momentum conservation, pressure integration, control surface, their numerical methods |
| [12. Second-order loads](12-second-order-loads.md) | QTFs: pairs, the quadratic part, the second-order potential, the indirect and direct methods, the floating body, the free surface, sub-problems, accuracy and cost, Newman's approximation, one wall, generalized modes |
| [13. Linear algebra, threads, determinism and memory](13-linear-algebra-and-threads.md) | The LU, the accuracy of complete results and the double-precision mode, threads and memory |
| [14. Geometry preprocessing and tolerances](14-geometry-preprocessing.md) | Panels and patches as the engine uses them, trimming, the tolerances of the geometry |

## Figures

The sketches of the chapters are drawn by `scripts/theory_figures.py` into `figures/`.
