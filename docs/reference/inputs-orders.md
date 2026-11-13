# Inputs: first and second order

## 1. First order

What the first-order solve computes, the items of `m.first_order` (Table 1).

*Table 1. The items of the first order.*

| Item | Default | Meaning |
|---|---|---|
| `radiation` | True | solve the radiation problems of the body item `modes` and of the generalized and pressure modes |
| `diffraction` | True | solve the incident-wave (diffraction) problem |
| `diffraction_formulation` | `"auto"` | unknown of the diffraction problem: `"scattered"` or `"total"` potential (both give the same forces); `"auto"` the total one when the model has a lid or a free-surface plate (the scattered one with low-order thin sheets), the scattered one otherwise; an explicit choice is checked (reference/inputs-rules §1) |
| `motion` | False | solve the motions (the RAOs; requires radiation and diffraction) |
| `field_pressure` | False | at the field points: the free-surface elevation ($z = 0$) and $p/(\rho g)$ below (theory §4.1, `.6p`); requires field points |
| `field_velocity` | False | the fluid velocity at the field points (theory §4.1): low order by the source formulation, higher order by differences of the potential (theory §4.2); requires field points; where it is available: reference/inputs-rules §2 |
| `body_pressure`, `body_velocity` | False | $p/(\rho g)$ and the fluid velocity on the body (theory §4.1): at the panel centroids (low order), at the parametric points of the patches (higher order) or at the body item `body_points`; low order the velocity by the source formulation (theory §4.2); where the velocity is available: reference/inputs-rules §2 |
| `field_points` | unset | the (N, 3) points of the field outputs in the sea, global coordinates (reference/inputs-model-objects §3); the points inside a tank are the tank's `field_points` |

## 2. Second order

The second-order loads, the items of `m.second_order` (and the items of the potential QTF's free surface, §3; Table 2).

*Table 2. The items of the second order.*

| Item | Default | Meaning |
|---|---|---|
| `drift_pressure_integration` | False | the mean drift force and moment by pressure integration over the body (theory §11.2), the six components of every body in body axes on its rigid coordinates, with internal tanks in the low order the tank fluid on its walls, with pressure surfaces the wetted surface only and the hydrostatic part in closed form (a warning), with generalized modes (low order, without pressure surfaces or tanks) their displacement in the terms and their generalized forces on their coordinates ([theory §12.15](../theory/12-second-order-loads.md#1215-generalized-modes)) |
| `drift_control_surface` | False | the mean drift by the momentum flux through the `control_surface` of every body (theory §11.3, `.7`), six components in body axes; every body needs its control surface |
| `drift_momentum_conservation` | False | the mean drift by momentum conservation in the far field (theory §11.1, `.8`), surge, sway force and yaw moment of the whole system in global axes. Each drift needs the diffraction and, when a body moves, `first_order.motion` (fixed bodies: `modes = []`); where each is available: reference/inputs-rules §2 |
| `drift_bidirectional` | False | the heading pairs of the drifts: False $(\beta, \beta)$ of every heading, True every pair $\beta_a \le \beta_b$ (reference/results); the bidirectional pressure and control-surface drifts are the diagonal of the difference quadratic QTFs of the heading pairs ([theory §12](../theory/12-second-order-loads.md)), without their moment about the moving rotation centre |
| `drift_momentum_conservation_angles` | `"auto"` | the number of Kochin angles of the drift by momentum conservation, 1..65536, or `"auto"`: chosen by the engine from the largest wave number and the radius of the bodies (theory §11.1, theory §11.4) |
| `qtf_quadratic_pressure_integration`, `qtf_quadratic_control_surface` | False | the quadratic part of the second-order force and moment of every body ([theory §12](../theory/12-second-order-loads.md)) at the difference and the sum frequency of every pair of the frequencies, on the coordinates (the six components of each body in body axes, the moment about its rotation centre, on its rigid coordinates), the products of the first-order solution: by pressure integration (theory §12.1, where that drift is available; the generalized forces of generalized modes on their coordinates) and over the control surface (theory §11.3: with the time derivatives over the free surface inside it and over the hull, which it takes from the pressure integration); the engine computes what they rest on and returns only what is asked for; needs the diffraction |
| `qtf_potential_direct`, `qtf_potential_indirect` | False | the force of the second-order potential: direct (the second-order potential solved; [theory §12.6](../theory/12-second-order-loads.md#126-the-direct-method)) and indirect (theory §12.5); the low order (the higher order refuses it); bodies fixed (`modes = []`) or floating with `first_order.motion` (their motions in the forcing; a body with modes or generalized modes needs `motion`), also with generalized modes (their motions in the forcing and their generalized forces on their coordinates, [theory §12.15](../theory/12-second-order-loads.md#1215-generalized-modes)); in the open sea or by one wall (`x_wall`, `y_wall`: the mirror problem of [theory §12.14](../theory/12-second-order-loads.md#1214-one-wall), the free surface the generated panels); not with `orthogonal_walls` or a channel, pressure surfaces, tanks, thin sheets, dipole dampers, damping lids or fixed dampers; with several bodies the boxes of their waterlines must not meet and $R$ must hold each with its free surface (§3) |
| `qtf_newman` | False | Newman's approximation of the difference-frequency QTF ([theory §12.13](../theory/12-second-order-loads.md#1213-newmans-approximation)) from every drift switched on (`drift_pressure_integration`, `drift_control_surface`, `drift_momentum_conservation`), at the equal headings; refused with no drift on |
| `qtf_newman_average` | `"geometric"` | `"geometric"`: $\operatorname{sign}(T_{ii})\sqrt{T_{ii} T_{jj}}$ (0 where the drifts differ in sign); `"arithmetic"`: $(T_{ii} + T_{jj})/2$ |
| `qtf_bidirectional` | False | the heading pairs of the QTFs: False $(\beta, \beta)$ of every heading, True every pair $\beta_a \le \beta_b$ |
| `qtf_difference`, `qtf_sum` | True, True | the frequencies of the QTFs: the difference and the sum frequencies (`qtf_sum = False` for the slow drift alone: the potential part then solves no radiation problems at the sum frequencies); both False with a QTF switched on is refused |

## 3. Free surface of the potential QTF (items of `second_order`)

The free surface the force of the second-order potential integrates
([theory §12.8-§12.9](../theory/12-second-order-loads.md#128-the-free-surface-inside-the-inner-radius)), in zones: the panelled zone
(from the waterlines to the inner radius), the quadrature zone (annuli of Gauss points), the outer
circle (a line integral) and beyond it the far field (a series of angular orders). Its items are
items of `m.second_order`: the switch `free_surface_method` chooses the panelled zone, the items
prefixed `free_surface_` belong to its values (the others kept unused), and the six quadrature
items are taken by every value:

```python
m.second_order.update(free_surface_method="cells", free_surface_panel_size=1.5)   # m
m.second_order.quadrature_annuli = 2
```

*Table 3. The items of the free surface of the potential QTF.*

| Item | Meaning | Default |
|---|---|---|
| `free_surface_method` | the panelled zone: `"panels"`, generated by the engine (a square grid about the waterlines with triangles to them, spokes and rings beyond, each panel integrated by its centroid), `"cells"`, generated ($4 \times 4$ Gauss points a cell), or `"mesh"`, given (`free_surface_mesh`, `free_surface_ranges`, `free_surface_quadrature_points`) | `"panels"` |
| `free_surface_mesh`, `free_surface_ranges` | `"mesh"`: the panelled zone given as a mesh of panels in $z = 0$, global coordinates, from the bodies' waterlines outwards (`ranges` unset: the whole mesh); the circle about the global origin through its farthest vertex is the inner radius; any panels mesh (`m.meshes.load_fdf(name, path, symmetry=...)` reads only the panels of an FDF file) | `mesh` none (needed), `ranges` unset |
| `free_surface_panel_size` | `"panels"`, `"cells"`: the size (m) of the generated panels or cells; `"auto"`: a third of the shortest combined wavelength $2\pi/(2k + k_\Omega)$ of the frequencies, at most $a/4$ | `"auto"` |
| `free_surface_inner_radius` | `"panels"`, `"cells"`: $R$ (m) of the generated panelled zone about the centre of the body origins (more than twice the waterlines' distance from it for the panels); it must exceed the radius of the bodies' wetted meshes about the centre (beyond it the fields are outgoing series); `"auto"`: $4a$, $a$ the largest distance from the centre to a waterline | `"auto"` |
| `free_surface_quadrature_points` | `"mesh"`: the Gauss points per side of each panel of the given mesh, 1..8 ($n \times n$ on its bilinear map); 1: the centroid with the area | 1 |
| `quadrature_annuli` | the annuli of the quadrature zone from the inner radius to the outer circle (inner radius + annuli $\times$ step); 0: none | 0 |
| `quadrature_radius_step` | the radial width (m) of each annulus; `"auto"`: two of the shortest wavelengths | `"auto"` |
| `quadrature_radial_nodes` | the Gauss-Legendre nodes across each annulus | 12 |
| `quadrature_azimuthal_nodes` | the equally spaced nodes at each radius of the annuli: on the circle, or a half circle or a quadrant with one or two symmetry planes of the whole model; `"auto"`: from the outer radius and the shortest combined wavelength | `"auto"` |
| `outer_circle_segments` | the equal segments of the line integral around the outer circle (counted as the azimuthal nodes); `"auto"`: the $4M_f + 8$ points of the far-field fit ($M_f = \max(M, \lceil k R_{OC} \rceil + 15)$, $R_{OC}$ the radius of the outer circle) | `"auto"` |
| `expansion_order` | the highest angular order $M$ of the far-field series beyond the outer circle (disturbances, radiation potentials or Green function, incident waves), 1..160; `"auto"`: $\lceil k R_{OC} \rceil + 15$, at most 160 ($R_{OC}$ the radius of the outer circle, $k$ the largest wavenumber of the frequencies and the pair frequencies) | `"auto"` |

With `free_surface_method = "mesh"` the panels must not overlap the bodies, its farthest vertex
must lie beyond the waterlines, `free_surface_mesh` is needed, and it needs a potential QTF
(`second_order.qtf_potential_direct` or `qtf_potential_indirect`).
The generated panels need closed waterlines, and are refused for a waterline too coarse for them
(a corner so far beyond the middles of its sides that the grid would need more than 50 steps, or
more than the panel size takes when that is more: give finer waterline sides, or a mesh) and for a
waterline inside another (a moonpool; a value error: they cover the water outside the waterlines
only, and a free-surface mesh can cover the water inside); the cells need a waterline of every
body (the panels take submerged bodies). The zones as the solve integrates them (no solve; the
frequencies decide the auto values): `m.panels(role="free_surface")` (the panelled zone's
panels, `Panels.role` `"free_surface"`, body $-1$; the cells have none) and
`m.free_surface_points(zone)` (an `hb.FreeSurfacePoints`, the `points` (n, 3) and `weights` (n,),
read-only, of `"panelled"`, `"quadrature"` or `"outer_circle"`; reference/queries §5). These
queries check the free surface's rules, not those of the orders or the frequencies; a zone whose
auto values the frequencies decide needs a frequency between 0 and $\infty$ (a value error without
one): the panelled zone of generated panels or cells with the auto `free_surface_panel_size`, the
quadrature zone with annuli and the auto `quadrature_radius_step` or `quadrature_azimuthal_nodes`,
the outer circle with the auto `quadrature_radius_step` (with annuli) or `outer_circle_segments`.
With one wall (`x_wall`,
`y_wall`) the zones are generated panels about the centre of the body origins projected onto the
wall, for the waterlines of the bodies and of their mirror images, and cover the whole plane
([theory §12.14](../theory/12-second-order-loads.md#1214-one-wall)): the points beyond the wall are among them.
