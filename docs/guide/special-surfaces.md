# Special surfaces

Real structures have parts that an ordinary wetted surface does not describe well: heave plates a
few centimetres thick, perforated screens, air chambers, moonpools, the narrow gap between two
hulls. HydroBEM has a dedicated kind of surface for each. This page explains what each one is for
and how to add it. Each is shown on its own; which combine with which is in
[reference/inputs-rules §2](../reference/inputs-rules.md#2-supported-combinations).

The surfaces that move with a body are children of the body, given in its body axes and included
in its forces. The surfaces that stay at rest, the damping lids and the fixed porous plates, are
lists of the model, given in global axes and part of no body's forces.

The examples use the model `m` and the functions `cylinder_panels` of
[Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder) and `disc_panels` of
[Meshes and bodies](meshes-and-bodies.md#3-one-mesh-several-surfaces). Several of them need a hull
with a moonpool, a vertical cylinder of radius 2 m and draft 1 m with a central opening of radius
1 m. Its mesh is made of four parts: the outer wall, the inner wall (its panels reversed so that
the normals point into the moonpool), the bottom ring, and a disc in the free surface inside the
moonpool. The examples look at the hydrodynamic coefficients only, so the motions, which would
need the mass properties of every body, are switched off:

```python
m.first_order.motion = False
outer = cylinder_panels(2.0, 1.0, n_bottom=0)               # the wall only, no bottom
inner = cylinder_panels(1.0, 1.0, n_bottom=0)[:, ::-1]       # normals into the moonpool
ring = disc_panels(2.0, -1.0, inner_radius=1.0, n_radial=4)
surface = disc_panels(1.0, 0.0, n_radial=4)                  # the water inside the moonpool
mp = np.concatenate([outer, inner, ring, surface])
m.meshes.add("moonpool", kind="panels", vertices=mp, symmetry="xz_and_yz")
n1 = len(outer) + len(inner) + len(ring)
hull_ranges = [(1, n1)]                                      # the wetted surface
surface_ranges = [(n1 + 1, n1 + len(surface))]               # the disc in the moonpool
```

## 1. The interior surface

The integral equation of the panel method fails at certain frequencies, the *irregular
frequencies*, which are the natural frequencies of the water that would fill the inside of the
hull up to the waterline. Near them the added mass, damping and exciting forces show sharp,
unphysical spikes. The larger the hull, the lower these frequencies, so on large structures they
fall among the wave periods of interest.

An *interior surface*, a surface in the waterplane inside the hull, removes them. The body item
`interior_surface` chooses how it is made:

- `"detect"` (the default): the elements of the body's wetted surface that lie in $z = 0$ become
  the interior surface; if there are none, the body has no interior surface;
- `"waterline"`, `"graded"` or `"projected"`: the engine generates it from the waterline of the
  body. `"waterline"` triangulates the waterplane and keeps holes such as moonpools; `"graded"`
  divides it into a fan of panels graded towards the waterline; `"projected"` projects the
  downward-facing wetted panels onto $z = 0$;
- `"mesh"`: given elements, with the items `interior_surface_mesh` and `interior_surface_ranges`;
- `"none"`: no interior surface.

```python
m.bodies["cylinder"].interior_surface = "waterline"
lid = m.bodies["cylinder"].panels(role="interior_surface")
print(len(lid.areas), lid.areas.sum())      # the generated panels and their area, m^2
```

The interior surface changes nothing at the other frequencies, so it is worth having on every
floating body that pierces the free surface. How it works and the evidence that it does is in
[theory §5](../theory/05-irregular-frequencies.md); the methods of generation are in
[reference/inputs-bodies §2.6](../reference/inputs-bodies.md#26-interior-surface-irregular-frequencies).

## 2. Thin plates

A heave plate, a strake or a baffle is often only centimetres thick, while its panels are a metre
wide. Meshed as a closed body with a panel on each face, it puts two nearly coincident surfaces
into the equations, which lose their accuracy. HydroBEM treats such a plate as a surface of zero
thickness instead, a *thin sheet* or *dipole*, moving rigidly with its body. Its elements are
given with `mesh` and `ranges`, either of the body's mesh or of a mesh of their own:

```python
m.meshes.add("plate", kind="panels", vertices=disc_panels(1.5, -0.8), symmetry="xz_and_yz")
m.bodies["cylinder"].dipoles.add("heave_plate", mesh="plate")
```

Which way the normals of a sheet point does not matter. A body may also consist of sheets only,
such as a submerged plate: it then has no `mesh` of its own. The mean drift by pressure
integration is not available with thin sheets; the drift by momentum conservation is. The theory
is in [theory §9.5](../theory/09-special-surfaces.md#95-thin-dipole-sheets), and the example
`examples/cylinder_with_heave_plate/run_thin_plate.py` compares a thin and a thick heave plate.

## 3. Porous plates

A perforated plate lets some water through and damps the flow. HydroBEM models it as a porous
sheet whose pressure drop is proportional to the flow through it, with a *damping factor*
$\varepsilon$ in m/s: $\varepsilon = 0$ lets the water pass freely, as if the plate were absent,
and a large $\varepsilon$ makes the plate solid. A porous plate that moves with a body is a *dipole
damper* of the body. Unlike a thin sheet, a dipole damper takes elements of the body's own mesh.
Here the heave plate of the spar of
[Meshes and bodies](meshes-and-bodies.md#3-one-mesh-several-surfaces), the elements 193 to 288 of
its mesh, is made porous, the solid plate being left out meanwhile:

```python
spar = m.bodies["spar"]
spar.dipoles["plate"].enabled = False
spar.dipole_dampers.add(mesh="spar", ranges=[(193, 288)], damping_factor=0.5)   # m/s
```

A porous plate fixed in space, such as a screen in a basin, is a *fixed damper* of the model. It
is given in global axes, by its `mesh` and `ranges`. Here it is a screen
2 m below the cylinder, in place of the spar's porous plate:

```python
spar.dipole_dampers.clear()
spar.dipoles["plate"].enabled = True
m.meshes.add("screen", kind="panels", vertices=disc_panels(1.5, -2.0), symmetry="xz_and_yz")
m.fixed_dampers.add(mesh="screen", damping_factor=0.5)        # m/s
m.fixed_dampers.clear()
```

The condition on a porous plate depends on the frequency, and it has no limit at $\omega = 0$ or
$\omega = \infty$: the added mass at the limits is not available with porous plates, nor is the
mean drift by momentum conservation ([theory §9.5](../theory/09-special-surfaces.md#porous-sheets-dipole-dampers)).

## 4. Air cushions and chambers

The water under an air cushion, or inside the chamber of an oscillating water column, is pressed by
the air above it. A *pressure surface* is a horizontal surface on which the pressure is
prescribed by *pressure modes*. Each mode is a shape of the pressure over the surface (uniform, or
the sign of $x$ or of $y$) and adds one coordinate to its body: a pressure head in metres. A
spring or damper on the mode, for example the stiffness of the compressed air or the damping of a
turbine, is given in N/m and N s/m:

```python
c = m.bodies.add("chamber", mesh="moonpool", ranges=hull_ranges, position=[10.0, 0.0, 0.0])
air = c.pressure_surfaces.add("air", mesh="moonpool", ranges=surface_ranges)
air.pressure_modes.add("uniform", shape="uniform", additional_damping=1.0e3)
print(m.counts().coordinates)         # six per body, plus one for the pressure mode
```

The pressure coordinate then appears among the coordinates of the results, with its own added
mass, damping, exciting force and response. The elevation of the surface is
`results.bodies["chamber"].pressure_surfaces["air"].elevation_radiation` and
`.elevation_diffraction`. A pressure surface is not available on a trimmed body. The theory,
with the two conventions of its restoring, is in
[theory §9.3](../theory/09-special-surfaces.md#93-prescribed-pressure-surfaces-air-cushions-chambers);
the example `examples/floating_owc_chamber` solves a floating water column with its air chamber
and turbine.

## 5. Plates on the free surface

A light piston floating on the water column of a wave energy device, a floating mat of zero
draft, or a rigid cover of a moonpool are rigid plates lying in the free surface with water on
one side only. HydroBEM models them as *free-surface plates*: massless rigid plates in $z = 0$ that
move with every mode of their body. A cover fixed to the hull is a plate of the hull's body:

```python
cover = m.bodies.add("covered", mesh="moonpool", ranges=hull_ranges, position=[0.0, 10.0, 0.0])
cover.free_surface_plates.add(mesh="moonpool", ranges=surface_ranges)
```

A plate that floats apart from the hull, such as the piston of a water column, is a body of its
own made of the plate alone, with no `mesh`. Its rigid heave, roll and pitch are the motions of
the plate. A plate has no mass and no volume of its own: the inertia is that of its body (zero
for a massless piston, whose inertia is then its added mass), and its area adds to the
waterplane of its body.

```python
m.meshes.add("piston", kind="panels", vertices=disc_panels(1.0, 0.0, n_radial=4),
             symmetry="xz_and_yz")
p = m.bodies.add("piston", modes=["heave", "roll", "pitch"], position=[0.0, -10.0, 0.0],
                 mass=0.0, center_of_gravity=[0.0, 0.0, 0.0])
p.free_surface_plates.add(mesh="piston")
```

With a plate, the diffraction problem is solved for the total potential by default, which is
more accurate under a plate; with low-order thin sheets in the model the default stays the
scattered potential. The mean drift by momentum conservation is available with every
plate. The mean drift by pressure integration, its quadratic QTFs and Newman's approximation of it
are available when every edge of a plate lies on a wall: a cover on its own body's moonpool wall,
or a piston sliding in a tube of another body or moved by a generalized mode. They are refused for
a plate with an edge bordered by free surface (a mat, a dock, a floating disc), and together with
pressure surfaces. The potential QTF (direct and indirect, in the low order) is available for the same plates: the free
surface ends at the plates, so a covered moonpool or a tube with its piston has one outer waterline;
take the generated panels or a free-surface mesh that leaves the plates out. With a plate edge
bordered by free surface the potential QTF is refused for good: its second-order forcing has no
finite integral at such an edge. The drift through a control surface and its QTFs are available
for the plates of the body that the control surface holds, a cover on its own moonpool wall, by
the same rule of the edges; its free surface ends at the hull's outer waterline. A piston that is
a body of its own lies inside the control surface of its hull, and a control surface must hold its
body alone, so that case is refused (use the pressure or the momentum drift). The
theory is in [theory §9.7](../theory/09-special-surfaces.md#97-rigid-plates-in-the-free-surface),
and the examples `examples/floating_owc_cover` and `examples/dock` solve a covered water column
and a floating dock.

On a trimmed hull a plate cannot be given as a mesh: the pitch or roll would tilt it out of the
water plane, and its edge must fit the waterline that the trim makes. The engine generates it
instead, from a point inside the waterline loop it covers (in the body axes of the trimmed hull):
the plate is the water inside that loop, triangulated like the interior surface, its edges on
the waterline. A trimmed hull is given whole, above the water too, so here the walls of the
moonpool hull reach 0.5 m above the water and the disc in its moonpool is left out. Pitched by
1.5 degrees, its moonpool is covered by a plate generated from the point at its centre:

```python
outer = cylinder_panels(2.0, 1.5, n_bottom=0) + [0.0, 0.0, 0.5]          # 0.5 m above the water
inner = (cylinder_panels(1.0, 1.5, n_bottom=0) + [0.0, 0.0, 0.5])[:, ::-1]
mp_full = np.concatenate([outer, inner, ring])
m.meshes.add("moonpool_full", kind="panels", vertices=mp_full, symmetry="xz_and_yz")
hull = m.bodies.add("pitched", mesh="moonpool_full", trim=[0.0, 1.5, 0.0], modes=["heave", "pitch"],
                    position=[-10.0, 0.0, 0.0])
hull.free_surface_plates.add(inside=(0.0, 0.0))                   # a cover of the moonpool
```

A piston that floats apart from the trimmed hull is a body of its own made of a generated plate,
which names the hull with the item `body`. Placed where the hull is, it is generated in the
hull's stored symmetry form. Here it takes the place of the cover:

```python
hull.free_surface_plates.clear()
p = m.bodies.add("pitched_piston", modes=["heave"], position=[-10.0, 0.0, 0.0],
                 mass=0.0, center_of_gravity=[0.0, 0.0, 0.0])
p.free_surface_plates.add(inside=(0.0, 0.0), body="pitched")      # the piston in the moonpool
```

A body made of plates only cannot itself be trimmed.

## 6. Damping lids

In a narrow gap between two hulls, or in a moonpool, the potential flow predicts resonant wave
heights far larger than those measured, because it has no viscous losses. A *damping lid* is a
patch of the free surface that adds a dissipation, with a damping factor $\varepsilon$ in m/s,
and brings the resonance down to a realistic height. A lid does not move and is part of no body's
forces. It is either a mesh of its own in global axes, or elements of a body's mesh placed with
that body (`placed_with`). Here the lid takes the place of the air chamber of §4 on the water in
the moonpool, since an element has only one role in a body:

```python
m.bodies["chamber"].pressure_surfaces.clear()
m.lids.add("moonpool_lid", mesh="moonpool", ranges=surface_ranges, placed_with="chamber",
           damping_factor=0.05)                                   # m/s
```

The elevation under the lid is in `results.lids["moonpool_lid"]`. With a lid the diffraction
problem is solved for the total potential by default (the scattered one with low-order thin
sheets in the model), and the mean drift by momentum conservation is not
available. The condition on the lid and the meaning of its damping factor are in
[theory §9.1](../theory/09-special-surfaces.md#91-physical-damping-lids).

## 7. Field points

The free-surface elevation, pressure and velocity at points in the water are computed at *field
points*, given in global axes in `m.first_order.field_points`. A point at $z = 0$ gives the
elevation; a point below gives the pressure as $p/(\rho g)$. The outputs are switched on with
`field_pressure` and `field_velocity`:

```python
m.first_order.update(field_points=[[5.0, 0.0, 0.0], [5.0, 0.0, -2.0]],
                     field_pressure=True)
print(m.classify_points([[5.0, 0.0, 0.0], [0.5, 0.0, -0.2]]).status)   # fluid, inside_body
```

A point inside a body, on its surface or inside its waterplane is refused; `m.classify_points`
gives the verdict on any points before a solve. The results are in `results.field_points`, one
row per point in the order given. The points inside an internal tank belong to the tank
([Tanks and generalized modes](tanks-and-generalized-modes.md#1-internal-tanks)). The rules for
field points are in
[reference/inputs-model-objects §3](../reference/inputs-model-objects.md#3-field-points-first_orderfield_points-a-tanks-field_points).
