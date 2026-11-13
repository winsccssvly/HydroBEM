# Tanks and generalized modes

The six rigid modes describe a body only as long as everything in it moves together. Liquid cargo
or ballast sloshes in its tanks, and a long hull or a mat bends in the waves. This page shows how
to add internal tanks, whose liquid HydroBEM solves for, and generalized modes, which add motions
of any shape to a body.

The examples use the model `m` and the function `cylinder_panels` of
[Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder).

## 1. Internal tanks

A tank is a volume of liquid inside a body, with its own free surface and no connection to the sea.
When the body moves, the liquid sloshes and pushes back on the walls; near the natural frequencies
of the sloshing this can change the motions of the body a great deal. HydroBEM solves the flow in
each tank as a problem of its own, driven by the motion of the walls, and adds the tank's added
mass, damping and restoring to those of its body. A tank receives no exciting force, since the
waves of the sea do not reach it.

The walls of a tank are elements of the body's mesh, with their normals pointing into the tank's
liquid. The tank also has:

- `density_ratio`: the density of its liquid relative to that of the sea (1 by default, 0 for an
  empty tank);
- `surface_level`: the height of its free surface in body axes, in m; by default the top of its
  walls;
- `field_points`: points inside the tank, in body axes, where the elevation (on the free surface)
  or the pressure (below it) of the sloshing is computed.

As an example, a floating cylinder of radius 5 m and draft 4 m carries a cylindrical tank of
radius 2 m, filled to 1 m below the waterline with a liquid of density 0.85 times that of the sea.
The tank walls are the panels of a smaller cylinder, lowered by 1 m and with their corners
reversed so that their normals point into the tank:

```python
outer = cylinder_panels(5.0, 4.0)
tank_walls = (cylinder_panels(2.0, 2.5) - [0.0, 0.0, 1.0])[:, ::-1]
m.meshes.add("tanker", kind="panels", vertices=np.concatenate([outer, tank_walls]),
             symmetry="xz_and_yz")
n = len(outer)
b = m.bodies.add("tanker", mesh="tanker", ranges=[(1, n)], position=[30.0, 0.0, 0.0],
                 mass="displaced", center_of_gravity=[0.0, 0.0, -2.0],
                 radii_of_gyration=[2.5, 2.5, 3.0])
b.tanks.add("ballast", mesh="tanker", ranges=[(n + 1, 2 * n)], density_ratio=0.85,
            field_points=[[1.5, 0.0, -1.0]])
print(b.hydrostatics().mass)            # the displaced mass less the liquid in the tank, kg
```

With the mass `"displaced"`, the mass of the body is the displaced mass less the liquid of its
tanks, as above. The outputs at the tank's points are switched on like those of any field point,
with `first_order.field_pressure`, and after a solve they are found under the tank:

```python
m.first_order.field_pressure = True
r = m.solve()
eta = r.bodies["tanker"].tanks["ballast"].pressure_radiation    # [F, points, columns]
print(eta.shape)
```

They are given per unit motion of each radiation column, since the liquid of a tank moves only
because its walls do. The elevation in waves is the sum over the columns weighted by the
motions, as shown for pressures in [Reading results](results.md#21-pressures-elevations-and-other-outputs).

Thin baffles inside a tank are added to the tank, not to the body: `tank.dipoles` for solid
baffles and `tank.dipole_dampers` for perforated ones
([Special surfaces](special-surfaces.md#2-thin-plates)). The theory of the tanks is in
[theory §10.2](../theory/10-generalized-modes-and-tanks.md#102-internal-tanks).

A deck over part of a tank, or a lid that closes it, is a plate of the tank's free surface:
`tank.free_surface_plates`, elements of the body's mesh or a mesh of their own lying at the
tank's `surface_level`, massless and rigid, moving with the body
([Special surfaces](special-surfaces.md) for the plates of the sea). The liquid under the plate
cannot rise there, so the plate's area no longer reduces the restoring; a tank covered all over is
a closed container whose liquid moves with the walls, its added mass the liquid's mass at every
frequency. The cover of the tank above, a disc of 2 m radius in a mesh of its own at the tank's
free surface: the bottom panels of a cylinder of draft 1 m, the last 16 x 6 panels of
`cylinder_panels`:

```python
cover = cylinder_panels(2.0, 1.0)[-16 * 6:]          # the disc at z = -1
m.meshes.add("cover", kind="panels", vertices=cover, symmetry="xz_and_yz")
b.tanks["ballast"].free_surface_plates.add(mesh="cover")
```

A plate on a tank's liquid belongs to that tank; a plate of the sea or a separate body floating
on it (a floating roof) is refused. The example
`examples/fish_tank_cylinders` solves floating tanks of zero wall thickness with the sloshing
inside.

## 2. Generalized modes

A *generalized mode* adds a motion of any shape to a body: the bending or torsion of a hull, the
deflection of a mat, the hinged motion of two parts, or the motion of a part of a device. Each
generalized mode is a coordinate of its own, after the six rigid ones, with its own radiation
problem, its own row in the equations of motion and its own response.

The shape of a mode is its displacement $(u, v, w)$ at each point of the body, given in pieces.
Each piece covers a range of elements of the body's mesh (all of it by default) and gives $u$,
$v$ and $w$ as polynomials in the body coordinates $x$, $y$ and $z$. A polynomial is a list of
terms, each a row `[c, px, py, pz]` for the term $c\, x^{p_x} y^{p_y} z^{p_z}$. The vertical
bending of the cylinder above, for instance, with $w = 0.01 x^2$:

```python
bend = m.bodies["tanker"].generalized_modes.add("bend", parity="heave")
bend.pieces.add(w=[[0.01, 2, 0, 0]])
```

`parity` names the rigid mode whose symmetry the generalized mode shares, so that the engine can
use the symmetry of the mesh: the shape $x^2$ is even in $x$ and in $y$, like heave.

The structure behind a mode, its mass, stiffness and damping, is not known to the panel method and
is given by the user. The body items `mode_mass`, `mode_stiffness` and `mode_damping` are square
matrices over the six rigid and the generalized coordinates of the body; their rigid block stays
zero, since the rigid inertia comes from the mass properties:

```python
k = np.zeros((7, 7))
k[6, 6] = 1.0e5                        # the stiffness of the bending, N/m
mm = np.zeros((7, 7))
mm[6, 6] = 50.0                        # its modal mass, kg
m.bodies["tanker"].update(mode_stiffness=k, mode_mass=mm)
r = m.solve()
print(r.coordinates[-1].name, abs(r.bodies["tanker"].motion[:, 0, 6]))
```

The hydrostatic restoring of a generalized mode is computed only in part; what it leaves out, such
as the gravity of the structure's own mass, belongs in `mode_stiffness`. The details are in
[theory §10.1](../theory/10-generalized-modes-and-tanks.md#101-generalized-modes), and every item
in [reference/inputs-bodies §2.5](../reference/inputs-bodies.md#25-generalized-modes).
Generalized modes also enter the mean drift by pressure integration and the second-order loads
of the low order, which give the generalized force of each mode
([theory §12.15](../theory/12-second-order-loads.md#1215-generalized-modes)).
