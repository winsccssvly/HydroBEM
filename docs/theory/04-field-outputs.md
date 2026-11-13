# 4. Field outputs

Besides the forces, a solution yields the flow itself: the elevation of the free surface around
the bodies, the pressure and the fluid velocity at points in the water, and the pressure and
velocity on the hull. This chapter defines the quantities and their components
(§4.1) and then states how each method computes them and how accurately (§4.2).

## 4.1 Free-surface elevation, field pressure and velocity

### Elevation and pressure

In open water the free-surface elevation follows from the dynamic free-surface condition:

$$
\eta = -\frac{i\omega}{g}\, \phi \qquad \text{at } z = 0. \tag{4.1}
$$

Below the free surface the same quantity is the dynamic pressure head,

$$
\frac{p}{\rho g} = -\frac{i\omega}{g}\, \phi, \qquad p = -i\omega\rho\phi. \tag{4.2}
$$

One quantity therefore serves both, and it is continuous at $z = 0$.

### The components that are reported

The field outputs are reported by component (Table 1). There is no total combined with the
motions (no RAO-combined total).

*Table 1. The components of the field outputs.*

| Component | Potential |
|---|---|
| incident | $\phi_I$ |
| scattering | $\phi_S$ |
| fixed-body total | $\phi_I + \phi_S$ |
| radiation, per unit displacement of each solved mode $j$ | $i\omega\phi_j$ |

The radiation component is per unit displacement, not per unit velocity: by (2.1) a unit
displacement in mode $j$ has the potential $i\omega\phi_j$, so its elevation (4.1) is

$$
\eta_j = -\frac{i\omega}{g}\, i\omega\phi_j = \frac{\omega^2}{g}\, \phi_j. \tag{4.3}
$$

The total of a moving body is the linear sum

$$
\text{total} = \text{diffraction} + \sum_k \xi_k\, \text{radiation}_k, \tag{4.4}
$$

over every radiation column, rigid, generalized and pressure coordinates alike, with the motions
$\xi$ of the solve (§3.1). Those motions depend on the mass, through $M$ and the weight terms of
$C$.

### The fluid velocity

With `field_velocity` (reference/inputs-orders) the fluid velocity $\nabla\phi$ is reported at the
field points, in global axes, with the same components: incident, scattering and total, and the
radiation $i\omega\nabla\phi_j$ per unit displacement. The two methods compute it differently
(§4.2):

- the low order from the source formulation, $v(x) = \sum_j \sigma_j \nabla_x \int_j G \,\mathrm{d}S$;
- the higher order by second-order differences of the potential of Green's identity, with the
  step $10^{-4} L$, one-sided in $z$ for a point in $z = 0$ **[C]**.

### Pressure and velocity on the body

With `body_pressure` and `body_velocity` (reference/inputs-orders) the same quantities are
reported on the body (Table 2).

*Table 2. Where the body outputs are evaluated.*

| Method | Points | Velocity |
|---|---|---|
| low order | the centroids of the wetted panels | from the source formulation, including the normal velocity |
| higher order | the parametric points of the patches | the surface gradient plus the normal velocity (§4.2) |
| both | the given `Body.body_points` (§4.2) | as above |

The body pressure is the dynamic pressure at the mean position of the point, without a hydrostatic
term:

$$
\frac{p}{\rho g} = -\frac{i\omega\phi}{g}. \tag{4.5}
$$

A point that moves with the body also changes its depth, and with it the hydrostatic pressure
$-\rho g z$. Its pressure adds this change:

$$
\frac{\Delta p}{\rho g} = -\sum_k \xi_k\, u_{z,k}(x), \tag{4.6}
$$

with $u_{z,k}$ the vertical displacement of coordinate $k$ at the point. For the rigid modes about
$R$ the sum of their displacements is

$$
\sum_{k=3}^{5} \xi_k u_{z,k} = \xi_3 + \xi_4 (y - y_R) - \xi_5 (x - x_R); \tag{4.7}
$$

for a generalized mode $u_{z,k}$ is its $w$, and for a pressure mode it is 0. The displacements are
those of the query `Model.mode_displacements` (reference/queries).

### Points that are refused

Some points have no meaningful value and are rejected with a reason:

- points at $z = 0$ on a body, on a lid or inside the waterplane of a body (for lid points the
  lid trace is used instead);
- submerged points inside a body. They are detected by the solid angle of the body's closed wetted
  surface together with its mirror image in $z = 0$, which is $\pm 4\pi$ inside.

In the low order a point closer to a panel's centroid than the panel's diameter is valid but warned
of, since the field there has the error of the constant-strength panels.

## 4.2 Velocities and pressure at points

### The potential at field points

The potential at a field point follows from Green's identity over the body, and over the interior,
lid and pressure surfaces as they enter their rows of the equation. Inside an internal tank it is
the tank's own problem (§10.2).

### Low order: the source formulation

For the velocities the low order also solves for a source distribution $\sigma$ on the body (Lee,
Newman & Zhu 1996):

$$
\phi(x) = \int \sigma(\xi)\, G(x, \xi) \,\mathrm{d}S_\xi, \tag{4.8}
$$

$$
\frac{\partial\phi}{\partial n_x} = -2\pi\sigma(x) + \int \sigma(\xi)\, \frac{\partial G(x, \xi)}{\partial n_x} \,\mathrm{d}S_\xi, \tag{4.9}
$$

where the derivative is taken at the field point $x$ on the body, along its normal into the fluid.
Equation (4.9) with the body condition is solved for $\sigma$ per symmetry character (§6.3), with
the same pair kernels as the potential. The velocity is then

$$
v(x) = \sum_j \sigma_j\, \nabla_x \int_j G \,\mathrm{d}S. \tag{4.10}
$$

The parts of the kernel are integrated as in Table 3.

*Table 3. The gradient of the kernel in the low-order velocity.*

| Part | Evaluation |
|---|---|
| Rankine part | exact (the principal value on the panel itself) |
| wave part | the centroid rule, from the field derivatives of the kernel |
| log correction of the pairs of §6.2 | its gradient; one $4 \times 4$ rule beyond 3 radii (a change below 1e-7) |

The same velocities serve the field points, the body panels and the pressure and control-surface
drifts. The waterline is the set of panel edges with both ends in $z = 0$; each has one point, at
its midpoint, with the potential of its panel. The source formulation is not available with thin
sheets.

### Pressure panels in the free surface

A pressure panel in $z = 0$ coincides with its own free-surface image, which needs care.

Its own log term is integrated whole; the kernel's one-point value there is its regular part
(§6.2). This gives the horizontal velocity: the in-plane principal values of the Rankine and log
terms, and a regular part whose horizontal gradient vanishes at the centroid.

The vertical derivative of that regular part keeps a log singularity, in the notation of §6.2
$\partial(\mathrm{Ft} + 2\psi)/\partial Y = -\mathrm{Ft}$. The vertical velocity therefore comes
instead from the identity of the source rows of these panels (§9.3),

$$
\phi_z = K\phi + 4\pi\sigma, \tag{4.11}
$$

with $\phi$ the potential the sources carry (the scattered one for diffraction). The free-surface
condition then holds exactly. The horizontal velocity is the limit of a surface lowered towards
$z = 0$, which is linear in its depth.

### Higher order: differences of the potential **[C]**

The higher order differentiates the field potential numerically, which avoids the second
derivatives of the finite-depth kernel. With the step $s = 10^{-4} L$, $L$ the reference length of
the first body's mesh (reference/inputs-meshes), second-order central differences at
$x \pm s\, e_k$ give

$$
\frac{\partial\phi}{\partial x_k} \approx \frac{\phi(x + s\, e_k) - \phi(x - s\, e_k)}{2s}. \tag{4.12}
$$

A point in $z = 0$ has no water above it, so its vertical derivative is one-sided, from $z$,
$z - s$ and $z - 2s$:

$$
\frac{\partial\phi}{\partial z} \approx \frac{3\phi(z) - 4\phi(z - s) + \phi(z - 2s)}{2s}. \tag{4.13}
$$

In double precision the truncation error is about 1e-8 and the rounding error about 1e-12,
relative, far below the accuracy targets.

### Higher order on the body

On the body the higher order uses the B-spline solution directly: the potential is its value, and
the velocity its surface gradient plus the normal velocity. They are evaluated at the midpoints of
the $(k_u + 1) \times (k_v + 1)$ subdivisions of every potential panel of the wetted and pressure
patches.

### Given body points

At given body points the two methods differ:

- **Higher order:** the closest point of the wetted and pressure patches, found from $8 \times 8$
  parametric seeds per instance and Gauss–Newton in $(u, v)$ clamped to the patch.
- **Low order:** the inverse-distance mean of the `body_point_neighbors` nearest centroids
  (reference/inputs-bodies §1.1); a point on a centroid takes its value.

The number $k$ of neighbours was chosen against the higher order on the hemisphere
([validation: constants](../validation/cases/constants.md)). No $k$ is best on every mesh
(Table 4).

*Table 4. The error of the low-order body points on the hemisphere, relative to the best number
of neighbours $k$ on each mesh.*

| Panels per quadrant edge | Best $k$ | $k = 3$ | $k = 4$ |
|---|---|---|---|
| 12 | 3 | 1 | 1.7 |
| 24 | 12 | 1.3 | 2.3 |
| 48 | 3 | 1 | 1.2 |

One neighbour errs four to five times more than three. The default 4, the centroids that surround
a point on a quadrilateral mesh, stays within 2.3 times the best $k$ on these meshes. The error
falls with the mesh more than with $k$.

### Points inside a body

A field point inside a body is detected by the solid angle of §4.1. A low-order field point closer
to a centroid than the panel diameter is valid but warned of (§4.1).
