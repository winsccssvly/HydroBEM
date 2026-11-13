# Mean drift and second-order loads

The linear results describe forces at the frequencies of the waves. A moored structure also feels
forces at second order in the wave amplitude: a steady *mean drift* force, which pushes it
downwave, and forces at the difference and the sum of every two wave frequencies. The slowly
varying difference-frequency loads excite the low natural frequencies of a mooring system; the
sum-frequency loads excite the high natural frequencies of a tension-leg platform. This page shows
how to compute the mean drift by its three methods, which one to trust, and how to compute the
quadratic transfer functions (QTFs) of the second-order loads.

All second-order loads are switched on by items of `m.second_order`. They are computed from the
first-order solution, so the diffraction problem must be solved, and a body that moves needs its
motions (`first_order.motion = True`); a fixed body has `modes=[]`. The examples continue the
floating cylinder of [Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder).

## 1. Mean drift

HydroBEM computes the mean drift force by three methods, each switched on by its own item:

- `drift_momentum_conservation`: from the momentum that the waves radiated and scattered by the
  bodies carry away to infinity. It gives the horizontal force and the yaw moment on all bodies
  together, in global axes;
- `drift_pressure_integration`: by integrating the second-order pressure over the wetted surface
  of each body. It gives all six components of each body, in body axes about its rotation centre;
- `drift_control_surface`: from the momentum flux through a *control surface*, a closed surface
  around each body at some distance from it (§3). It also gives six components per body.

```python
m.environment.headings = [0.0, 30.0]
m.second_order.update(drift_momentum_conservation=True, drift_pressure_integration=True)
r = m.solve()
print(r.drift.heading_pairs)                          # (0, 0) and (30, 30)
print(r.drift.momentum_conservation[:, 0, 0].real)    # surge drift force, all bodies, N/m^2
print(r.drift.pressure_integration[:, 0, 0].real)     # surge drift force of the cylinder
```

The drift forces are per unit wave amplitude squared, in $\mathrm{N/m^2}$ (and $\mathrm{N\,m/m^2}$ for
moments). Their arrays
have the axes F (the frequencies), P (the pairs of headings) and the components. With the default,
P has one pair $(\beta, \beta)$ per heading, whose drift is real. With `drift_bidirectional=True`
the drift is computed for every pair of headings $\beta_a \le \beta_b$, the mean force of two
waves of the same frequency from different directions; those coefficients are complex. The drifts
of a body are on its coordinates, like its exciting forces: the six rigid components, then the
generalized force of each generalized mode, and NaN where a method has no value. The drift of one
body is also found under the body, `r.bodies["cylinder"].drift.pressure_integration`.

### 1.1 Which drift to trust

The drift by pressure integration needs the fluid velocity on the body surface, which becomes
singular at sharp edges, such as the bottom rim of a truncated cylinder or the edges of a box. For
a floating body with such edges near its heave or pitch resonance, this drift converges very
slowly as the mesh is refined, and it can be far off. Use the drift by momentum conservation
there, which gives the surge and sway forces and the yaw moment of all bodies together, or the
drift through a control surface, which gives all six components of each body. Keep the drift by
pressure integration for bodies without sharp edges, for fixed bodies, or as a cross-check. The
three methods are compared in
[theory §11.3](../theory/11-mean-drift.md#113-mean-drift-force-and-moment-control-surface).

## 2. Comparing the methods

On a smooth body all three methods agree once the mesh is fine enough, so asking for two of them
in one solve is a cheap check. The example above already computed two:

```python
diff = r.drift.pressure_integration[:, 0, 0] - r.drift.momentum_conservation[:, 0, 0]
print(abs(diff) / abs(r.drift.momentum_conservation[:, 0, 0]).max())
```

A large difference near a resonance of a body with sharp edges is the behaviour described in
§1.1, not an error of the input.

## 3. The control surface

The drift through a control surface needs a surface around every body: the body item
`control_surface`. HydroBEM can generate a truncated vertical cylinder, the lower half of an
ellipsoid, or a surface that follows the waterline of the body at a given distance (`"automatic"`),
or it takes a mesh, for example one read from a CSF file with `m.meshes.load_csf`. The control
surface must enclose its body alone, and its panels should be about as fine as those of the body:

```python
m.bodies["cylinder"].update(control_surface="cylinder", control_surface_radius=2.0,
                            control_surface_depth=1.5, control_surface_inner_radius=1.0,
                            control_surface_panel_size=0.2)       # m
m.second_order.drift_control_surface = True
r = m.solve()
print(r.drift.control_surface[:, 0, 0].real)
```

The cylinder of these examples shows the behaviour of §1.1. At the period of 2 s, close to its
heave resonance, the drift by pressure integration on this coarse mesh is about twice the drift by
momentum conservation, while the drift through the control surface agrees with the momentum
within a few per cent.

Here `control_surface_inner_radius` is the radius of the body's waterline, inside which the
free-surface part of the control surface ends. The items of each kind of surface are listed in
[reference/inputs-bodies §2.7](../reference/inputs-bodies.md#27-control-surface). The control
surface is not available with tanks, air cushions, generalized modes or porous plates, nor with
thin plates in the low order (the higher order has it with thin plates).

## 4. Second-order loads (QTFs)

In an irregular sea, every two wave components of frequencies $\omega_a$ and $\omega_b$ give a
second-order force at the difference frequency $\omega_a - \omega_b$ and at the sum frequency
$\omega_a + \omega_b$. The coefficients of these forces are the *quadratic transfer functions*, or
QTFs. HydroBEM computes the QTFs of every pair of the model's frequencies in two parts, which are
added to give the total:

- the *quadratic part*, the products of the first-order solution, by pressure integration
  (`qtf_quadratic_pressure_integration`) or through the control surface
  (`qtf_quadratic_control_surface`);
- the *force of the second-order potential*, the flow that those products force at the
  difference and the sum frequency, by a direct or an indirect method (`qtf_potential_direct`,
  `qtf_potential_indirect`).

```python
m.second_order.reset("drift_control_surface")
m.bodies["cylinder"].control_surface = "none"      # a control surface needs its drift
m.second_order.update(qtf_quadratic_pressure_integration=True, qtf_potential_direct=True)
r = m.solve()
q = r.qtf.quadratic.pressure_integration.difference   # [F, F, P, D]
p = r.qtf.potential.direct.difference                 # [F, F, P, D]
total = q + p
print(total.shape)
```

The arrays have the axes F and F (the two frequencies of the pair), P (the pairs of headings, by
`qtf_bidirectional` as for the drift) and the coordinates D. The element `[a, a]` of the
difference QTF of the quadratic part is the mean drift. The force of the second-order motions
themselves is not included. With `qtf_sum=False` only the difference frequencies are computed,
which is all that a study of slow drift needs and saves the work of the sum frequencies.

The two methods of the quadratic part and the two of the potential part each check one another.
The control surface converges faster with the mesh than the pressure integration, for the reason
of §1.1. The direct and the indirect method of the potential part discretize the same problem in
two ways and agree to within their discretization error; asking for both in one solve costs little
more than one.

The force of the second-order potential is computed by the low-order method, in the open sea or by
one wall (`x_wall`, `y_wall`), for fixed bodies and for floating bodies with their motions, also
with generalized modes. It is not available with thin or porous plates, tanks, air cushions or
damping lids. The theory is in [theory §12](../theory/12-second-order-loads.md), and the example
`examples/second_order_bottom_cylinder` compares it with a published semi-analytic solution.

## 5. Newman's approximation

At the difference of two close frequencies the QTF is close to the mean drift. Newman (1974)
approximates the whole difference-frequency QTF from the drifts alone, with no second-order
computation. HydroBEM forms it from every drift that is switched on (and refuses it with none):

```python
m.second_order.update(qtf_quadratic_pressure_integration=False, qtf_potential_direct=False,
                      qtf_newman=True)
r = m.solve()
newman = r.qtf.newman.pressure_integration.difference   # [F, F, H, D], real
print(newman[0, 1, 0, 0], r.drift.pressure_integration[0, 0, 0].real)
```

The approximation is real and defined for equal headings only. By default an off-diagonal value is
the geometric mean of the two drifts, $\operatorname{sign}(\bar F_a) \sqrt{\bar F_a \bar F_b}$
(zero where the drifts differ in sign); with `qtf_newman_average="arithmetic"` it is their mean
$\tfrac{1}{2}(\bar F_a + \bar F_b)$. Newman's approximation stands for the whole
difference-frequency QTF, not for a part to be added to the others
([theory §12.13](../theory/12-second-order-loads.md#1213-newmans-approximation)).

## 6. The free surface of the second-order potential

The force of the second-order potential depends on an integral over the whole free surface around
the bodies. HydroBEM evaluates it in zones: a zone of panels from the waterlines out to an inner
radius, an optional ring of quadrature points beyond it, and the far field beyond that, where the
flow is a series of outgoing waves. By default the engine generates the panels and chooses their
size and the inner radius from the frequencies and the size of the bodies.

The default inner radius is four times the largest distance from the centre of the bodies to a
waterline, and the default panel size a third of the shortest wavelength of the products, at most
a quarter of that distance. The sum frequencies vary faster over the free surface than the
difference frequencies, so they are the first to need smaller panels: if the sum-frequency loads
matter, repeat the solve with a smaller `free_surface_panel_size` and compare. The zones the solve
will integrate can be seen without solving:

```python
m.second_order.update(qtf_newman=False, qtf_potential_direct=True,
                      free_surface_panel_size=0.1)              # m
fs = m.panels(role="free_surface")
print(len(fs.areas), fs.areas.sum())        # the panels of the inner zone, m^2
```

The panels can also be given as a mesh (`free_surface_method="mesh"`, with `free_surface_mesh`),
for example one read with `m.meshes.load_fdf`. Every item of the free surface is in
[reference/inputs-orders §3](../reference/inputs-orders.md#3-free-surface-of-the-potential-qtf-items-of-second_order),
and the method and its accuracy in
[theory §12.8](../theory/12-second-order-loads.md#128-the-free-surface-inside-the-inner-radius) to
[§12.12](../theory/12-second-order-loads.md#1212-accuracy-and-cost).
