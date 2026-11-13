# Water, waves and options

Besides its bodies, a problem needs the water they float in, the waves that meet them, and a
choice of what to compute and by which method. These are set on four objects that every model has:
`m.environment` (the water and the waves), `m.first_order` (what the linear solve computes),
`m.options` (how it is solved) and `m.second_order` (the mean drift and second-order loads, on
their own page). This page goes through the first three.

The examples continue [Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder)
with the model `m` of the floating cylinder.

## 1. The water

The water is described by its depth, its density and the acceleration of gravity. By default the
water is infinitely deep, $\rho = 1000\ \mathrm{kg/m^3}$ and $g = 9.80665\ \mathrm{m/s^2}$:

```python
m.environment.update(depth=30.0, density=1025.0)     # m, kg/m^3
m.environment.depth = np.inf                          # deep water again
```

The sea bed is the plane $z = -h$, flat and horizontal. A gravity written in a mesh file is kept
with the mesh but not used; the solve takes `environment.gravity`.

## 2. Walls and channels

A body near a quay or in a test basin feels the reflections from the walls. The item `domain`
adds vertical walls that the water cannot pass:

- `"open_sea"`: no walls (the default);
- `"x_wall"`: a wall in the plane $x = 0$, the water on the side $x > 0$;
- `"y_wall"`: a wall in the plane $y = 0$, the water on the side $y > 0$;
- `"orthogonal_walls"`: both walls, the water in the corner $x > 0$, $y > 0$;
- `"channel"`: a channel between the walls $y = \pm W/2$, with `channel_width` $W$, in water of
  finite depth.

With walls, the incident wave includes its reflections. In a channel the waves travel along the
channel only, so the headings are 0 or 180 degrees. Real basin walls do not reflect perfectly;
`channel_reflect` (between 0 and 1, 1 by default) scales the reflections of the channel walls,
and 0 gives the open sea of the same depth.

```python
m.environment.update(domain="channel", channel_width=12.0, depth=5.0, headings=[0.0, 180.0])
print(m.route())
```

The width belongs to the channel only. To return to the open sea, clear it together with the
domain, since a channel width without a channel is refused:

```python
m.environment.update(domain="open_sea", channel_width=None, depth=np.inf)
```

A body that lies wholly in the plane of a wall is a *wavemaker*: a flap or piston in the wall,
whose radiation problem alone is solved
([theory §10.3](../theory/10-generalized-modes-and-tanks.md#103-wavemakers-in-the-walls)). The
walls and their reflections are described in
[theory §1.2](../theory/01-boundary-value-problem.md#walls-and-channel).

## 3. Frequencies and headings

The waves are regular waves of unit amplitude, given by their frequencies and their headings. The
frequencies are given either as radian frequencies $\omega$ in rad/s (`omegas`) or as periods
$T = 2\pi/\omega$ in seconds (`periods`). The switch `frequency_input` says which list is read,
`"omegas"` (the default) or `"periods"`:

```python
m.environment.update(frequency_input="omegas", omegas=np.linspace(0.5, 4.0, 8))   # rad/s
m.environment.update(frequency_input="periods", periods=[2.0, 4.0, 8.0])          # s
```

The headings are the directions in which the waves travel, in degrees counter-clockwise from the
$+x$ axis. They are needed whenever the diffraction problem is solved:

```python
m.environment.headings = [0.0, 45.0, 90.0]
```

The results keep the frequencies in the order given (`results.omegas`) and the headings as given
(`results.headings`). The wave number $k$ of a frequency, the root of
$\omega^2 = g k \tanh(k h)$, is `hb.wavenumber(omega, depth)`.

### 3.1 Zero and infinite frequency

Time-domain models need the added mass at zero and at infinite frequency. They are requested by
putting the values 0 and `np.inf` among the frequencies (or, with periods, the period 0 for
$\omega = \infty$):

```python
m.environment.update(frequency_input="omegas", omegas=[0.0, 1.0, 2.0, np.inf])
r = m.solve()
print(r.added_mass_zero_frequency[2, 2], r.added_mass_infinite_frequency[2, 2])
print(r.omegas)                          # the ordinary frequencies: 1.0 and 2.0
```

At the two limits no waves are radiated, so only the added mass is defined there, and it is
returned on its own. The results at the other frequencies, the *ordinary frequencies*, are
unchanged by the request. The limits are explained in
[theory §1.3](../theory/01-boundary-value-problem.md#13-frequency-limits-omega--0-and-omega--infty).

## 4. What the first order computes

The linear (first-order) problem has two parts: the *radiation* problems of the body moving in
each of its modes in calm water, which give the added mass and damping, and the *diffraction*
problem of the body held still in the incident waves, which gives the exciting forces. The
switches of `m.first_order` choose what is solved and what is reported (Table 1).

*Table 1. The switches of the first order.*

| Item | Default | Computes |
|---|---|---|
| `radiation` | True | the radiation problems: added mass and damping |
| `diffraction` | True | the diffraction problem: the exciting forces |
| `motion` | False | the motions (RAOs) from the mass properties; needs both problems |
| `body_pressure`, `body_velocity` | False | the pressure and the fluid velocity on the body surface |
| `field_pressure`, `field_velocity` | False | the free-surface elevation, pressure and velocity at given points in the water ([Special surfaces](special-surfaces.md#7-field-points)) |

```python
m.environment.update(frequency_input="periods", periods=[2.0, 4.0, 8.0])
m.environment.headings = [0.0]
m.first_order.update(motion=True, body_pressure=True)
r = m.solve()
print(r.body_points.pressure_diffraction.shape)    # frequencies, panels, headings
```

The exciting force is computed by integrating the pressure of the diffraction problem over the
body. It can also be found from the radiation problems by the Haskind relation; both are in the
results, and the body item `excitation_method` (`"diffraction"` or `"haskind"`) chooses which one
drives the motions. The item `diffraction_formulation` chooses whether the diffraction problem is
solved for the scattered or the total potential; its default `"auto"` picks the better one for
the surfaces of the model. The meaning of every item is in
[reference/inputs-orders §1](../reference/inputs-orders.md#1-first-order).

## 5. The low-order and higher-order methods

The option `m.options.higher_order` chooses the method
([Meshes and bodies](meshes-and-bodies.md#1-what-a-mesh-is)). With `False`, the default, every
body needs a panel mesh; with `True`, every body needs a patch mesh. The higher order is more
accurate for a given number of unknowns on smooth bodies. The low order has the wider range of
features, among them the force of the second-order potential, which the higher order does not
compute; a few combinations, such as the fluid velocity near thin plates, are available in the
higher order only.

```python
m.meshes.analytic("cylinder_exact", "circular_cylinder", [8, 4, 3, 3, 3, 4],
                  radius=1.0, draft=0.5)         # the exact cylinder, as in Meshes and bodies
m.bodies["cylinder"].mesh = "cylinder_exact"
m.options.higher_order = True
r = m.solve()
m.options.higher_order = False
m.bodies["cylinder"].mesh = "cylinder"
```

Two more options concern accuracy. `logarithmic_integration` (True by default) integrates the
logarithmic singularity of the wave part of the flow analytically near the free surface; it
should stay on. `log_tolerance` (1e-4) chooses, for the low order, the pairs of panels close
enough to need it. Both are explained in
[theory §6.2](../theory/06-low-order-method.md#62-the-logarithmic-singularity-of-the-wave-kernel).
The options for threads and memory are on the page
[Performance, threads and progress](performance.md).

## 6. Which features combine

Every feature works in deep and finite water, for one or several bodies, with symmetry and with
the frequency limits, unless stated otherwise. Not every feature combines with every other: the
mean drift by momentum conservation, for example, needs the open sea without damping lids, and
the mean drift by pressure integration is not available with thin plates. Before a long solve,
`m.check()` and `m.route()` check the model against the engine without solving it, and a
combination that is not supported raises `hb.UnsupportedProblemError` naming it. The full table of features, methods,
domains and outputs is in
[reference/inputs-rules §2](../reference/inputs-rules.md#2-supported-combinations).
