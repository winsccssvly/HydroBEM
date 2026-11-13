# Test plan

Two kinds of evidence support every feature: independent mathematics (A) and
identities and invariants (B); the regression baselines (R) detect a change of the results of
the features they cover, the second-order loads among them (§R).
The convergence studies this plan calls validation, the published references over their whole
range and the sensitivity of the fixed constants are shown with figures in the
[validation report](README.md), one page per case.

## A. Independent mathematics

*Table 1. The tests against independent mathematics: the id, the test and its threshold.*

| Id | Test | Threshold |
|---|---|---|
| A1 | Rankine panel integrals vs high-precision adaptive quadrature, field points on/near/far from the panel, edges, vertices | $\le$ 1e-5 relative (float32 path) |
| A2 | Green function (infinite, finite depth, derivatives) vs the implementer's own arbitrary-precision evaluation of the integral representation over the full domain | theory §8 |
| A3 | Dispersion roots (propagating and evanescent) | full double precision |
| A4 | **[P]** Floating lower hemisphere of radius $a$ in infinite depth vs Hulme (1982), tables 1 and 2, at $Ka = \omega^2 a/g = 0.5$, 1.0, 2.0 (those rows of `tests/data/hulme1982_full.txt`). With $m_0 = 2\pi\rho a^3/3$, compare heave $A_{33}/m_0$ and $B_{33}/(\omega m_0)$ with $A^{(0)}$ and $B^{(0)}$, and surge $A_{11}/m_0$ and $B_{11}/(\omega m_0)$ with $A^{(1)}$ and $B^{(1)}$. Use at least three successively refined low-order meshes. The wider published range ($0.1 \le Ka \le 10$) is the test of the irregular-frequency removal (`test_walls_and_interior_surfaces.py`): on a graded mesh of 16 divisions per quadrant edge, without an interior surface the difference at the irregular frequencies $Ka = 4$ and 7 is at least 5e-3 (about 1e-2), with the projected interior surface at most 2e-3 there and at most 7e-3 over the range (the discretization of 16 divisions). The whole range on 48 divisions ($\le$ 1e-3) and its convergence over 16, 32 and 48 divisions are validation. | Each of the four quantities $\le$ 1e-3 of the maximum absolute published value among the compared records at the finest mesh, converging with refinement |
| A5 | **[P] [C]** Truncated vertical cylinder in finite depth vs an independently implemented eigenfunction-matching reference derived from a published formulation (e.g. Bhatta and Rahman 1995 for heave and Bhatta 2003 for surge; Yeung 1981 may also be used). The reference must be separate from the production BEM and demonstrate truncation convergence to $\le$ 1e-5. Its sources give their results as figures only, so no published number anchors it: the reference rests on the published formulation, its truncation convergence and its agreement with the BEM. The test compares two meshes (24 and 32 panels a quarter circle) with the reference: the difference at 32 and the order of convergence between them. The finest mesh (48), the convergence of three successively refined low-order meshes (monotone for every quantity) and Richardson's extrapolation with the order observed for each quantity are validation, run when the method changes **[C]**. | Each quantity $\le$ 3e-3 at 32 panels, converging between 24 and 32 at an order between 1.5 and 2.2 (test); $\le$ 2e-3 at 48, monotone and $\le$ 1e-3 after mesh extrapolation (validation) |
| A6 | Zero/infinite-frequency limits of a sphere/hemisphere vs closed forms | $\le$ 1e-3 on 36 panels a quadrant edge ($O(h^2)$) |
| A7 | Hydrostatics of boxes, cylinders, spheres: closed forms (exact for higher order; constant-panel discretization error $-A h^2/12$ per direction for low order second moments, exact for volumes and area) | $\le$ 1e-12 relative of the expected discrete value |
| A8 | Inertia transport and $K_R = T^{\mathsf{T}} K_P T$ vs direct algebra | $\le$ 1e-12 of the peak (double round-off) |
| A9 | Second-order potential load ([theory §12](../theory/12-second-order-loads.md)): the Stokes transform of the motion terms (§12.7) on a hemisphere against the Hessian of incident waves; Bessel $J_n$, $Y_n$ against mpmath values (orders to 150), the Wronskian and the sum rule; $e^x K_m(x)$ (orders 0, 1, 5, 30) against SciPy; the free-surface points between two rectangles (their area and second moments); the transform of a deformation that is not rigid (with its divergence), the gradient of a generalized mode against central differences and the first-order surface element (Nanson) | the transform with the waterline extrapolation within 2e-3 (without it more than 0.1; without the deformation terms more than 5 %), the gradient 1e-8, the surface element 1e-15; the mpmath values 1e-12, the Wronskian 1e-9 (relative to $\lvert J_{n+1} Y_n \rvert + \lvert J_n Y_{n+1} \rvert$), the sum rule 1e-12, $K_m$ 1e-12; the area and second moment 1e-5 (the square root at a tangent radius), no point inside a rectangle |
| A10 | Free-surface plates ([theory §9.7](../theory/09-special-surfaces.md#97-rigid-plates-in-the-free-surface), `test_free_surface_plates.py`): at $\omega = 0$ a plate is a piston in the rigid plane $z = 0$ (Rayleigh's piston), a disc against $A_{33}(0) = (8/3)\rho a^3$ and a square against the self-integral of the square in closed form; the dock, a disc of zero draft in finite depth, against eigenfunction matching (`scripts/gen_cylinder_reference.py`); at $\omega = \infty$ a disc against half of Lamb's disc, $(4/3)\rho a^3$ in heave and $(8/45)\rho a^5$ in pitch | the disc at $\omega = 0$ 1e-3 and 2e-4 on 8 and 16 rings a quadrant, falling more than fourfold; the square 3e-3 and 1e-3 on 8 and 16 panels a side of the quadrant (low order), 6e-4 and 2e-4 on 2 and 4 cubic panels (higher order); the dock $A$ 2e-4 and $B$ 4e-3 on 16 rings a quadrant; Lamb's disc 3.5e-2 on 32 graded rings a quadrant, falling with the rings, and 3e-3 extrapolated in the panel size (the square-root edge of a sheet), the higher order 1e-4 (heave) and 2e-4 (pitch) on 4 radial spans graded to the edge and 3e-5 on 6 |
| A11 | A fully covered tank is a closed container ([theory §10.2](../theory/10-generalized-modes-and-tanks.md#102-internal-tanks), `test_tank_free_surface_plates.py`): its added mass in surge, sway and heave is the fluid mass $\rho_T V_T$ at every frequency and both limits, and its damping zero | low order 7e-2 (the cover's edge on the walls, $O(h)$), surge and sway 1.2e-2 at the finite frequencies, the damping 4e-3 of the peak; the higher order on flat patches 1e-5 (the rigid flow is exact; the float32 storage) |

## B. Identities and invariants

The identities hold exactly in the continuous problem; the tests bound what the discretization
and single precision leave. The bound is relative to the peak of the compared quantity. Where a
feature needs a larger bound than the general one, Table 2 gives it with its cause; the
identities with a bound for many features are listed after Table 2, one feature per line.

*Table 2. The identities and invariants: the general bound and the larger bounds with their
causes.*

| Identity | General bound | Larger bounds and their causes |
|---|---|---|
| Energy: the damping from the far field or the Kochin functions equals $B$ (Newman 1977), $B_{jj} = (\rho \omega K_e/2k) \int \lvert b_j \rvert^2 \,\mathrm{d}\theta$, any depth | $\le$ 2e-3 (low order, from the Haskind force over the headings on 24 panels a quadrant edge; 1e-3 through field points after extrapolation in the radius) | higher order from the Kochin functions, infinite and finite depth: 1e-3 for surge and heave, 3e-3 for pitch; the work of a wavemaker equals the flux of its waves within 2e-3 |
| Mean drift (theory §11.1): for a fixed body without dissipation the cross term equals the energy form (optical theorem); a symmetric body in head seas has no sway or yaw drift; a translated body keeps its drift, its yaw moment changes by $x_0 \bar{F}_y - y_0 \bar{F}_x$; a free body held by a stiff spring has the fixed body's drift | $\le$ 1e-5, the float32 panel integrals (1e-6 for the zero sway and yaw drift and for the body held by a stiff spring) | — (the pressure drift against the momentum drift with the mesh, the higher against the low order with pressure surfaces and the convergence in the Kochin angle count are validation, run when the method changes) |
| Newman's approximation (theory §12.13): the diagonal is the drift, the matrix symmetric, the geometric mean squared the product of the two drifts of one sign (0 where not), the arithmetic their mean; a drift computed for it alone is not returned | exact (the same numbers), 1e-12 relative | — |
| Control-surface drift (theory §11.3): the horizontal drift equals the momentum drift, no component depends on the surface; the free-surface part enters the vertical force and horizontal moments only; the two forms of the line term have the same vertical terms; several bodies, each in its own control surface: their horizontal drifts sum to the momentum drift and two mirrored bodies have mirrored drifts; a surface reaching another body or holding a damping lid is refused | low order 3 % (the fields at the panel centroids of the surface); higher order 1e-5, the second form 1e-3, the radii 1e-3; the exact parts bit-identical; two bodies: the sum 3 % of the peak, the mirror 1e-5 | — |
| Quadratic QTFs (theory §12.1): the difference coefficient of a wave with itself is the pressure drift (both methods, two moving bodies, pressure surfaces with the closed forms, a tank); Hermitian (difference) and symmetric (sum) in the pair of waves; the heading pairs contain the equal headings | $\le$ 1e-12 | at equal frequencies the heading pairs follow the momentum drift of the pairs: 3 % at equal headings, 6 % between them on a low-order cylinder of $16 \times 8 \times 8$ panels per quadrant (the pressure drift's convergence; 1.3 % and 4.7 % observed, conjugated pairs 170 %) |
| The force of the second-order potential ([theory §12](../theory/12-second-order-loads.md)), each check on the basis it rests on: the difference coefficients zero at equal frequencies and Hermitian, the sums symmetric (identities of the definitions); the direct and the indirect method (theory §12.6) in one solve, one continuous problem; a floating body held by a stiff spring has the fixed body's coefficients and two fixed bodies mirror each other's (identities of the problem); the auto radius and size equal to those given; the rules of the inputs; the panels of an FDF file; a bottom-mounted cylinder ($h/a = 1$) by both methods against the semi-analytic solution of Kim & Yue (1990, table 3) | the symmetries $\le$ 1e-12; direct against indirect: differences 2e-2, sums 5e-2 (a wrong term gives $O(1)$); the spring 1e-5; the mirror 1e-2 (the triangles next to the waterlines are not mirrored); auto 1e-9; Kim & Yue 1.5 % | — (the convergence in $R$, the discretizations against each other and the zones beyond $R$ are validation, run when the method changes: $R = 3$ to 10 m against 12 m within 1 % of the largest coefficient, the heave of the test cylinder within 0.2 %, theory §12.12, [validation: cylinder](cases/cylinder.md)) |
| The force of the second-order potential by one wall ([theory §12.14](../theory/12-second-order-loads.md#1214-one-wall)): a body by an x or a y wall, fixed and floating, both methods, against the open-sea problem of the body and its mirror image (a second body) under the wave and its image, the bidirectional coefficients of the four heading pairs summed; the wall's coefficients are the even part $(A + S B)/2$ of that problem ($A$ the body's, $B$ the image's, $S$ the mirror image of a force and moment), the same panels in both; the motions the sum of both headings'; two walls and a channel refused | the even part 1e-5 (1.3e-6 observed, the float32 first-order fields); against $A$ 1e-2 (2e-3: the triangles next to the waterlines are not mirror images); the motions 1e-5 | — (a half body on the wall: bodies lie strictly inside the fluid) |
| Second-order loads of generalized modes ([theory §12.15](../theory/12-second-order-loads.md#1215-generalized-modes)): the rigid modes given as generalized modes (the rigid body's mass and restoring) have the rigid body's pressure drift, quadratic QTFs and potential QTFs (both methods), fixed (held by a stiff spring), floating with the translations as generalized modes, by an x wall; with every rigid mode generalized the drift differs by the hydrostatic terms of the second-order rotation and the stretched surface element alone (computed from the panels) and the potential QTF in heave alone (head seas); the generalized rows of flexible modes are Hermitian and symmetric with their drift on the diagonal; a mode that tears the wetted surface open and the higher order are refused | the spring 1e-7 (potential 1e-5); floating 1e-12 (4e-16 observed: the same first-order solution); the hydrostatic terms 1e-12 (2e-15 observed); the symmetries 1e-12 | a cylinder with a bulging bottom and bending sides: the pressure drift against the momentum drift, surge and sway, 30 % on 8 panels a quadrant, falling with the panels (24 % and 11 % observed on 8 and 16, 5 % on 32); the potential QTF's generalized rows direct against indirect, differences 5e-3, sums 2e-2 (1.4e-3, 9e-3 observed) |
| The velocities of the source formulation with an interior surface equal those without it away from the irregular frequencies, also at the waterline (the interior free surface carries the vertical velocity of the exterior flow there, theory §5.2) | the body velocity 2.5e-3 of the peak on 12 panels a quadrant (1.7e-3 observed; 3.6 % with a rigid lid); 24 panels (1e-3, 5.3e-4 observed) is validation | — |
| Control-surface QTFs (theory §11.3): the difference diagonal is the control-surface drift; Hermitian and symmetric pairs; a control surface of radius 2 and one of radius 3 around a free cylinder give the same coefficients at all pairs (the time derivatives over the free surface inside the surface and over the hull) | the diagonal and the symmetries $\le$ 1e-12; the radii $\le$ 1e-4 (higher order), 2e-3 (low order): the discretization of the two surfaces (2.3e-6 and 5.1e-4 observed) | — |
| Control-surface drift of a body that is not wall-sided (theory §11.3): a free cylinder whose side flares to 1.3 times its radius at the waterline, the vertical force from the control surface against the pressure integration | the difference $\le$ 0.1 of the waterline term with the motions only, $(1/2) \rho g \int n_z/\sqrt{1 - n_z^2}\, \langle \zeta_b^2 \rangle \,\mathrm{d}l$, which exceeds the force itself | the pressure integration tends to the control-surface force with the mesh (within 1–2 % on $32 \times 16 \times 16$ panels per quadrant) |
| Total vs scattered diffraction formulations | $\le$ 2e-3 (low and higher order, thin-sheet patches; the two discretize different unknowns, the difference shrinks with the mesh) | 3e-3 with interior surfaces; 2e-2 with pressure surfaces |
| A wall equals the body and its mirror body; orthogonal walls four bodies | $\le$ 1e-6 (low order, infinite and finite depth) | 1e-5 higher order and a low-order body mirrored about the centre line of a channel (1e-6 the higher-order one: the float32 rounding of its stored coefficients); 2e-5 thin sheets (the float32 sums of the dipole rows in another order) |
| A symmetry-sector solve equals the full solve | $\le$ 1e-6 (low order, infinite depth) | 1e-5 low order in finite depth, higher order, the channel, lids, interior surfaces and pressure surfaces (1e-6 the higher order in a channel); 2e-5 thin sheets (as above); 2e-6 a box with a bending generalized mode (float32 sums, 8.7e-7) |
| Thread equivalence: 1, 2 and the threads of a test, frequency and inner | bit-identical (low and higher order, the limits, the fields, lids and interior surfaces with walls, and the regression problems with a channel, drift, tanks and a control surface; the QTFs: the quadratic, control-surface and Newman's approximation, the potential QTF by a wall, the second order of generalized modes (regression problems), and the potential QTF with a quadrature zone by both methods in `test_potential_qtf.py`) | — |
| Body permutation (several bodies) and translation and yaw placement | the permutation $\le$ 4 float32 ulps of the largest value (the results are stored in float32; 1.6e-7 found); translation and yaw placement $\le$ 1e-5 (the results transform with the rigid motion) | — |
| Winding reversal of a thin sheet leaves the forces unchanged; a dipole whose `mesh` names another mesh than the body's equals the same elements as dipoles on the body's mesh | $\le$ 1e-6 for the reversal, also of higher-order sheet patches (also the Haskind force with the jump term of the sheet; the same equations up to signs); $\le$ 1e-10 for the dipole of another mesh (the same equations) | — |
| A closed side of a higher-order sheet patch (the outer circle of an annulus in one patch) is split by its length like any side | one patch and four quarter patches give the same heave added mass within 1e-4; the heave-pitch and heave-roll coupling of the annulus $\le$ 1e-4 of the heave added mass | — |
| A damping lid with $\epsilon = 0$ equals no lid | $\le$ 1e-5 (float32 round-off; 1e-6 for several lids with walls); the lid trace at $\epsilon = 0$ equals the open-water elevation at the lid centroids within 1e-4 | — (the fall of the peak lid elevation and heave excitation with $\epsilon$ is validation) |
| Interior-surface results equal the plain results away from the irregular frequencies, and remove the spike at them | $\le$ 1e-3 | 2e-3 for a finite-depth added mass (discretization, $O(h^2)$); the generated surfaces against an explicit surface and against each other are validation ($\le$ 1e-3 at 16 panels a quadrant edge; the graded fan of a box 3e-3 on $8 \times 4 \times 4$ panels, 1e-3 on twice as fine) |
| RAO residual: the solved $\xi$ satisfies the motion equation, with the matrices of every coordinate (several bodies, higher order, tanks, generalized modes with `mode_mass`, `mode_stiffness`, `mode_damping`, pressure modes, the Haskind force, a low-order body with a full inertia tensor and additional stiffness) | $\le$ 1e-6 of the force | — |
| Partial radiation columns are subsets of the full solve; a diffraction-only solve gives the same excitation and fields as the full solve | $\le$ 1e-6 of the peak for the partial columns (the added mass); bit-identical for the diffraction-only solve | — |
| The frequency limits are approached by small and large finite frequencies (convergence, not substitution) | the error falls with the frequency | $A(\omega) \to A(\infty)$ to $\le$ 1e-2 at $\omega = 16$ rad/s on the test box (0.4 % observed), then limited by its panels per wavelength; $A(\omega) \to A(0)$ to $\le$ 1e-2 at $Ka = 10^{-3}$ in infinite depth ($O(Ka \ln Ka)$), $\le$ 1e-6 of the surge added mass at $\omega = 10^{-3}$ in finite depth ($O((kh)^2)$) |
| The Green function near its closed forms: the imaginary part of the infinite-depth kernel, $\mathit{Gi} = -2\pi e^{-Y} J_0(X)$ and $\mathit{Gi}_X = 2\pi e^{-Y} J_1(X)$; $F_X / X \to -F_{YY} / 2$ and $\mathit{Gi}_X / X \to \pi e^{-Y}$ on the vertical axis | $\mathit{Gi}$ and $\mathit{Gi}_X$ $\le$ 1e-7 of $2\pi e^{-Y}$; the axis quotients within $10^{-7} + X^2 \max(1, 2/Y^4)$ of their limits | the finite-depth quotient $G_R / R$ (`R_R`) within $10^{-6} + R^2$ of $G_{RR}$ (`RR`) on the axis |
| The radiation rows of the field-point file (`.6p`) of a moving body hold the normalized field values | $\le$ 1e-6 relative (the 7 significant digits of the file; 4.1e-7 found) | — |
| The channel in the higher order: the control-surface drift does not depend on the surface; without reflection the open sea | the control-surface drift $\le$ 1e-5 of the peak; the open sea $\le$ 1e-10 | — (the wall effect against the low order as both meshes refine is validation: 7e-3 at 32 panels per quarter, the higher order within 3e-3 between 6 and 8 spans) |
| Higher order with surfaces of meshes of their own: a lid in global coordinates, dipoles, a pressure and an interior surface as meshes of their own against the same elements of one mesh; a lid follows its body; flat with B-spline patches | bit-identical; the moved body 1e-10; the mixed kinds 1e-10 | — |
| The geometry file .hbg: every mesh kind written and read back (also through its object) to the same items; a mesh from GDF and SPL and through the geometry file give the same results; the discretization from one place; the format, version, members, kinds and JSON checked | the items equal, the results bit-identical; every refusal names the file, the patch and the member (the JSON error its line and column) | — |
| Rational B-spline patches: points on the exact surface; the derivatives the differences of the points, both evaluators equal, a common factor of the weights and weights 1 no change; the volume and restoring of the cylinder and the hemisphere; the hemisphere against Hulme (1982) and the analytic sphere | the surface 1e-15; the derivatives 1e-8 (1.7e-10 found), the evaluators equal, the factor and weights 1 1e-14; the volume and restoring 1e-12; Hulme 3e-4 of the largest tabulated value (four digits), the analytic sphere 1e-4 | — (the cylinder against the analytic cylinder, 2e-6 at 4 spans, is validation) |

Further identities and closed forms, each of its own test file:

- trimmed waterline (`test_trimmed_waterline.py`): a pitched and rolled box keeps its volume
  (1e-9), the pitched cylinder of patches is exact (1e-12), a trimmed body equals its trimmed
  meshes as one mesh, a generated interior surface covers the waterplane once (1e-6);
- the submerged disk against the closed form $(8/3)\rho a^3$ in unbounded fluid (`test_thin_sheets.py`):
  3.5e-2 at 24 panels and 1e-2 after the extrapolation in the panel size (a square-root edge
  singularity: first-order convergence);
- tanks (`test_internal_tanks.py`): the reduction of $C_{33}$ by $\rho_T g A_T$ and the change of $C_{44}$
  against the discrete closed form (1e-12), the mass `"displaced"`;
- files (`test_model_files.py`, `test_results_file.py`): a model and its results written and read
  back give the same bytes, and a model read back solves to the same bytes;
- disabled objects (`test_enabled.py`): a model with a disabled object gives the results of the
  model without it, byte for byte;
- point classification (`test_classify_points.py`) and the field (`test_field.py`): the verdicts
  of the field-point rules, the velocity as the gradient of the pressure, the submerged pressure
  continuing the elevation, the velocities beside a wall equal to those of the mirror body;
- free-surface plates (`test_free_surface_plates.py`, theory §9.7): a plate of the hull's body
  equals the hull and the same plate as a body of its own moving together (the rigid connection,
  1e-5); a plate by a wall equals its mirror pair and a channel without reflection the open sea
  (1e-5); the Haskind relation (2e-4) and the energy of the radiated wave (3e-4) of a disc; the
  two formulations and the symmetry sectors (1e-5); a wetted disc of small draft tends to the
  plate (1.2e-2 at the smallest draft); the momentum drift of a disc conserves energy (1e-5 of the
  energy form); the pressure drift of a cover and of a piston against the momentum drift (6e-2 and
  2.5e-2 on 16 panels a quadrant, falling with the panels); the control-surface drift of a cover
  against the momentum drift (8e-3 and 4e-3, falling) and independent of the surface (1.5e-3); a
  generalized piston mode against a piston body (1e-10); the potential QTF of a cover of the hull
  against a cover body (1e-5), Hermitian and symmetric (1e-12);
- free-surface plates generated on a trimmed hull (`test_generated_plates.py`, reference/inputs-bodies
  §2.4): at zero trim the generated plate is the plate given as a mesh (bit-identical); the
  generated plate fills its loop (its area 1e-12, its edges on the waterline); a small trim against
  a hand-made cover of the hull trimmed in the test (2e-2, and 1.5 times the difference without the
  pitch); on the pitched moonpool the rigid connection (1e-5), and the Haskind relation, the
  symmetry of $A$ and $B$ and the two formulations (low order 3e-2 and 1.5 times the error without
  the pitch, higher order 1e-3);
- the plates of a tank's free surface (`test_tank_free_surface_plates.py`, theory §10.2): besides
  A11, a half cover at $\omega = 0$ is the open tank in the modes that do not move it (1e-6), a
  vanishing cover tends to the open tank (2e-2), the tank with a plate is reciprocal (5e-3), and
  the plate's area leaves the free-surface restoring of the tank (1e-12);
- the second-order loads about the rotation centre R (`test_second_order_rotation_center.py`,
  theory §1.1, §12.2): the drifts by pressure integration and by the control surface, their
  quadratic QTFs and the potential QTF of a fixed body about R equal those about the origin turned
  by $M_R = M_O - R \times F$, plus the load of the rigid second-order translation $-H R$ (the
  hydrostatic load in the quadratic part, the radiation load in the potential part), within 2e-6
  of the peak (the float32 first-order outputs); the momentum drift does not depend on R (1e-2,
  the far-field sums at low frequencies); the pressure and the control-surface drifts about R
  agree in the horizontal force and the yaw moment (8e-2 on 32 panels around);
- the analytic shapes (`test_analytic_geometry.py`, `test_queries_by_owner.py`): their volume
  and restoring exact (1e-12).

The components without a public function are tested in their modules (`#[cfg(test)]`): the A9
items in `potential_qtf/` (`bessel.rs`, `body.rs`, `zones.rs`, and `far_field.rs`: the fields of
the quadrature zone from its circles) and `pressure_drift.rs`; the triangulation of regions with
holes in `polygon.rs` (area, orientation, holes, loops that meet, the Delaunay flips, the bridges
past other holes); the moonpool refusal of the generated free-surface panels in
`free_surface_panels.rs`; the parsing of the environment hooks in `env_hooks.rs`.

### Symmetry of A and B

$B \ge 0$ on the diagonal ($\ge -10^{-6}$ of the peak) in every case. The bound of the symmetry:

- general (low order, infinite and finite depth): $\le$ 1e-5;
- higher order: 2e-3 for $A$ (8 patches per quadrant);
- interior surfaces: 3e-3 (8 panels per quadrant edge: 1.1e-3 for $A$, 1.9e-3 for $B$);
- damping lids: 1.5 times the level of the same moonpool without the lid plus 1e-4 (the
  sharp-edged moonpool is as asymmetric without a lid; the fall with the mesh is validation);
- several lids and walls: 4 times the level of the same problem without lids;
- dipole dampers: 5e-3; the symmetric part of $B$ is positive semidefinite (eigenvalues $\ge -10^{-6}$ of
  the peak for one body: no energy is created; $-10^{-4}$ for two coarse hulls with a damper of the
  water and in the higher order);
- channel: 1.5 times the level of the same body in open water plus 1e-4 (the channel adds no
  error of its own); higher order within its bound of 2e-3 (the sway-heave coupling of the walls,
  $O(h^2)$).

### Haskind vs pressure excitation, at converged meshes

- general: $\le$ 1e-3 (low order: in infinite depth on 24 panels a quadrant edge, the decrease with
  refinement is validation; in finite depth on 12);
- higher order and pressure patches: 2e-3 (the patch quadrature at the waterline);
- higher-order thin-sheet patches: 5e-3;
- interior surfaces: 1e-2;
- fixed porous dampers and pressure modes: 2e-2 (the pressure integration over the sheet or
  pressure panels, whose velocity is singular at their edges); a damper of the body 1e-2;
- a wall: 2e-3 (the coarse test box, as in open water: 9.7e-4);
- damping lids: twice the level of the same moonpool with an undamped lid, or 2e-2 if larger;
- low-order thin sheets: 1e-2 (the coarse hull: 2.6e-3 without the sheet);
- channel: 1.5 times the level of the same body in open water plus 1e-4;
- the motions of the two forces on a coarse cylinder: 1.5e-2.

### Thin sheets, pressure surfaces and lids together

- Validation (not tests): a closed slab tends to the sheet: within 1.5e-2 (with a lid), 3e-2 (with a pressure surface)
  and 5e-2 (higher order) of the sheet at the thinnest; each halving of the thickness at most
  0.6 of the error.
- A lid at zero damping is no lid: within 1e-6.
- The scattered and the total formulation with a lid differ as without it: within 1.1 times their
  difference without the lid (the field and the lid elevation 1.5).
- Validation (not tests): the effect of a plate or a lid on a body is that of the higher order as the low order refines:
  the low order within 6e-2 (the plate) and 3e-2 (the lid) of the higher order at twice the
  panels, at most 0.6 of the coarser.

### Lids, thin sheets and dampers in a channel

- The far images of a sheet row are the derivatives of those of the body rows: within 1e-6 of
  central differences (1e-10 found).
- Without reflection the open sea: 1e-5 (low order), 1e-10 (higher order).
- The mirror body: 1e-5.
- A lid at zero damping is no lid: 1e-6 (1e-10 higher order); the formulations differ as without
  it.
- Validation (not tests): a closed slab tends to the sheet (also at $\omega = 0$ and $\infty$): within 1.5e-2 (low order) and 5e-2
  (higher order) of the sheet at the thinnest, each halving at most 0.6 of the error.
- Validation (not tests): the wall effect with a lid is that of the higher order as the low order refines: within 3e-2
  of the higher order at 8 panels per metre, at most 0.6 of the coarser.
- A damper at zero damping is none (1e-6), at large damping the sheet (1e-3: the $O(\omega/\epsilon)$ residual of
  the jump, 3.3e-4 at $\epsilon = 10^5$).
- The wall condition at the field points: the slope of the field at the wall $\le$ 1e-4 of $k$ times
  its peak (2e-4 with the sheet 1 m from the wall).

### Interior meshes

- **[P] [C]** Generated `projected` and `waterline` interior meshes lie at
  $z = 0$ with upward normals, preserve every outer waterline and moonpool hole,
  and cover the exact interior waterplane.  For procedural boxes, cylinders
  and annular bodies, their area and first moments agree to $\le$ 1e-12 relative
  with independently computed values of the exact polygonal waterline defined
  by the input mesh.  The two generators may use different valid
  panelizations.  Each result converges under refinement to the result from an
  equivalent explicitly selected or external interior mesh and differs by
  $\le$ 1e-3 at the finest tested mesh (validation, run when the method changes, not a unit
  test).
- **[P]** `projected` rejects a multi-valued vertical projection and
  `waterline` rejects open contours and contours that cross or touch; neither silently falls back
  to the other generator.

## R. Regression baselines

`tests/python/test_regression.py` solves the small problems of
`tests/python/regression_problems.py` and compares every output with
HydroBEM's stored result (`tests/data/regression/`). The problems cover:
- the low and higher order, finite depth and the limits;
- walls, the channel (also partly reflecting, and in the higher order with an interior patch) and a
  wavemaker in the walls;
- interior surfaces, lids, pressure surfaces, thin sheets and dampers, internal tanks;
- generalized modes, an analytic shape and a trimmed patch mesh;
- several bodies, field and body outputs;
- the pressure, momentum (also bidirectional) and control-surface drifts, and the Haskind
  excitation;
- the second order: the quadratic and control-surface QTFs with Newman's approximation, the
  potential QTF (both methods, a quadrature zone), the potential QTF by a wall, and the
  second-order quantities of generalized modes.

Each entry is held to 1e-5 of its own largest value over the first axis (the frequencies, or
the points), and at least to 1e-3 of the peak of its array, so that a small coefficient keeps
its own accuracy, with an absolute floor of 1e-14 (SI units) for an entry that is zero up to
round-off (the waterplane centroid of a symmetric body). The baselines detect changes; they are no reference of accuracy. After an
intended change they are regenerated by `scripts/gen_regression_baselines.py` and the reason
given in the commit. The README of the baselines records the processor and the linear-algebra
library they were written with (faer's kernels may differ in the last float32 bits between
processors).

## Error-handling tests

Every rule of reference/inputs-rules §1 has a test that the documented error type is raised before
any computation; unsupported combinations raise `UnsupportedProblemError`.
