# Inputs: lids, fixed dampers and field points

The lists `m.lids` and `m.fixed_dampers` hold what exists independently of the bodies, in global
coordinates, at rest and in no force; the field points are inputs of `m.first_order` and of the
tanks, the free surface of the potential QTF items of `m.second_order`.

## 1. Damping lids (`m.lids`, `DampingLid`)

- A dissipative patch of the free surface at global $z = 0$; its elements may be
  oriented upwards or downwards (downward is the convention of lid panels in the
  WAMIT format) **[B]** and are treated with upward normals; no mass,
  no modes, it does not move.
- A named object of the list `m.lids` with the items `mesh`, `ranges`, `placed_with` and
  `damping_factor` $\varepsilon \ge 0$ (m/s; theory §9.1): `m.lids.add(name, mesh=...)` a mesh in global
  coordinates; with `placed_with` the name of a body, elements of that body's mesh, placed with
  its position and yaw (a role of the body, reference/inputs-meshes §3; not of a trimmed body). The higher order takes
  a lid in global coordinates as patches of the first body, moved into its coordinates (reference/inputs-bodies §2; with
  symmetry flags the body on its symmetry planes without yaw; analytic patches only for a body at
  the origin without yaw).
- With a lid the diffraction problem is solved in the total formulation
  (`diffraction_formulation = "auto"`; the scattered one with low-order thin sheets).

## 2. Fixed dampers (`m.fixed_dampers`, `FixedDamper`)

Porous sheets fixed in space: `FixedDamper` objects with the items `mesh` and `ranges` (global
coordinates) and `damping_factor` $\varepsilon \ge 0$, the jump $-(i\varepsilon/\omega) \phi_n$, in no force (theory §9.5; $\varepsilon = 0$: no
jump, as no sheet; $\varepsilon \to \infty$: the rigid sheet, a dipole). A damper whose mesh is the mesh of a body at
the origin without yaw or trim takes elements of that body's mesh (a role of the body, reference/inputs-meshes §3); the
higher order has only these.

## 3. Field points (`first_order.field_points`, a tank's `field_points`)

- `m.first_order.field_points`: the points in the sea, an (N, 3) array of global points at $z = 0$
  (free-surface elevation) or below it and above the bottom (pressure), or unset. A point in the
  fluid domain only: not on a body, a lid or inside a water plane at $z = 0$, not inside a body
  below it (theory §4.1); `m.classify_points(points)` gives the verdict on any points.
- `tank.field_points` (`m.bodies[b].tanks[t]`, reference/inputs-bodies §2.2): the points inside that tank, an (N, 3)
  array in body axes, at or below the tank's `surface_level` (at it: the elevation); with a trim
  (reference/inputs-bodies §1.2) the body axes are those the trimmed hull is placed in, not its mesh coordinates: the
  points are not moved by the trim, and in global axes they are the body's position plus the
  point turned by its yaw; they get the
  outputs of the tank's radiation problem, with no incident or diffracted wave, and move with
  the tank: a rename or reordering of the body or the tank needs no change.
  `m.classify_points(points, tank)` gives the verdict on points in body axes inside the tank `tank`
  (a `Tank` object; a tank the solve does not take, disabled or of a disabled body: `"no_tank"`).
- The outputs follow the owners of the points (reference/results): those of the points in the
  sea are `results.field_points` (one row per point, in input order), those of a tank's points
  the tank's, `results.bodies[b].tanks[t]` (its points in body axes as given, the radiation of the
  tank problem). A message or location of a point names it in its owner's list: a point of the
  sea by its number in `first_order.field_points`, a point of a tank by its number in the tank's
  `field_points` and the tank (`field point 2 of bodies/hull/tanks/aft lies above the free surface
  of its tank`, the location with the tank's body). `m.counts()` counts the points in the sea
  (`field_points`) and in the tanks (`tank_points`). Points given without a field output
  (`first_order.field_pressure`, `field_velocity`) are refused, and so is a field output without
  points.
