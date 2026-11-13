# Meshes and bodies

A body in HydroBEM is defined by two things: its geometry, given as a mesh, and its properties as
a rigid body (where it is, how heavy it is and how it may move). This page shows how to make and
check meshes, how one mesh can carry several surfaces, and how to set up one or several bodies,
including a body at a trimmed waterline.

The examples continue [Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder):
the model `m` and the function `cylinder_panels` defined there.

## 1. What a mesh is

A *mesh* is the geometry of a surface, cut into *elements*. HydroBEM has two methods, and each
has its kind of element:

- the *low-order method* uses flat quadrilateral *panels* (a triangle is given by repeating a
  corner); the flow is taken as constant over each panel;
- the *higher-order method* uses curved *patches*: flat patches, B-spline or NURBS patches, or the
  exact patches of a few named shapes; the flow varies smoothly over each patch.

The low order is the usual choice and has the wider range of features; the higher order reaches a given
accuracy with far fewer unknowns on smooth bodies. The option `m.options.higher_order` chooses
the method ([Water, waves and options](environment-and-options.md#5-the-low-order-and-higher-order-methods)).

The corners of a panel (or the parameters of a patch) are ordered so that its normal points out of
the body into the water. A mesh may store one half or one quarter of a symmetric body and declare
the planes of symmetry with its item `symmetry`: `"yz"` for the plane $x = 0$, `"xz"` for the
plane $y = 0$, `"xz_and_yz"` for both (one quarter stored), or `"none"`. The engine then
reflects the stored part, and it also uses the symmetry to solve faster
([Performance](performance.md#3-symmetry)).

The elements of a mesh are numbered from 1 in the order they are stored, before the reflections.
These numbers are how a body or a surface chooses its part of a mesh (§3).

## 2. Making a mesh

Every mesh is an object of the list `m.meshes` and has a name. Table 1 shows the ways to make one.

*Table 1. The ways to make a mesh.*

| Call | Makes |
|---|---|
| `m.meshes.load(name, "hull.gdf")` | a low-order mesh from a GDF file |
| `m.meshes.load(name, "hull.gdf", higher_order=True, spl="hull.spl")` | a higher-order mesh from a GDF file of flat, B-spline or analytic patches, with its discretization from an SPL file; `panel_size=` or `discretization=` can replace the SPL file |
| `m.meshes.load(name, "hull.hbg")` | a mesh of any kind from HydroBEM's own geometry file |
| `m.meshes.add(name, kind="panels", vertices=v, symmetry=...)` | a low-order mesh from an array `v` of shape (number of panels, 4, 3) |
| `m.meshes.add(name, kind="flat_patches", ...)`, `kind="spline_patches"` | higher-order meshes from arrays |
| `m.meshes.analytic(name, shape, discretization, **dimensions)` | the exact geometry of a cylinder, sphere, ellipsoid, barge and a few other shapes, for the higher order |
| `m.meshes.load_csf(name, "cs.csf", body=...)` | a control surface for the mean drift ([Mean drift and second-order loads](second-order-loads.md#3-the-control-surface)) |
| `m.meshes.load_fdf(name, "fs.fdf")` | the panels of a free-surface file, for the second-order potential ([Mean drift and second-order loads](second-order-loads.md#6-the-free-surface-of-the-second-order-potential)) |

The reverse direction is `mesh.write`: it writes a panel, flat-patch or B-spline mesh as a GDF
file (with an SPL file for patches), or any mesh as a geometry file when the path ends in `.hbg`.
As an example, we write the cylinder of Getting started to a GDF file and read it back as a new
mesh:

```python
m.meshes["cylinder"].write("out/cylinder.gdf", overwrite=True)
m.meshes.load("cylinder_from_file", "out/cylinder.gdf")
print(m.meshes.names())
```

The file formats and every item of a mesh are described in
[reference/inputs-meshes](../reference/inputs-meshes.md).

For the higher order, the named analytic shapes are the quickest start. The discretization gives,
for every patch, the number of spans in the two directions of the patch, the order of the
B-splines that describe the flow, and two orders of Gauss quadrature; one row serves for all
patches:

```python
m.meshes.analytic("cylinder_exact", "circular_cylinder", [8, 4, 3, 3, 3, 4],
                  radius=1.0, draft=0.5)
```

## 3. One mesh, several surfaces

A body names its mesh with the item `mesh`, and it can take only part of the mesh with the item
`ranges`: a list of ranges of element numbers, `(first, last)`, both included. Without `ranges`
the body takes every element. This lets one mesh file hold a hull together with its heave plates,
tank walls or other surfaces, each surface naming its own elements; within one body, each element
can have only one such role.

To show this we need a second helper, which makes a flat ring or disc of panels at a given depth.
It also serves on later pages for plates, lids and covers:

```python
def disc_panels(radius, z, n_around=16, n_radial=6, inner_radius=0.0, up=False):
    """One quarter of a horizontal disc (or ring) at height z; the normals point down,
    or up with up=True."""
    phi = np.linspace(0.0, np.pi / 2, n_around + 1)
    r = np.linspace(radius, inner_radius, n_radial + 1)

    def p(radius, angle):
        return [radius * np.cos(angle), radius * np.sin(angle), z]

    panels = [[p(r[i], phi[j]), p(r[i + 1], phi[j]), p(r[i + 1], phi[j + 1]), p(r[i], phi[j + 1])]
              for i in range(n_radial) for j in range(n_around)]
    return np.array([q[::-1] for q in panels] if up else panels)
```

A spar of radius 1 m and draft 4 m with a heave plate of radius 2 m below it then makes one mesh
of two parts. The first 192 elements are the spar (16 around, 6 down the side and 6 across the
bottom); the next 96 are the plate:

```python
spar = cylinder_panels(1.0, 4.0)
plate = disc_panels(2.0, -4.5)
m.meshes.add("spar", kind="panels", vertices=np.concatenate([spar, plate]),
             symmetry="xz_and_yz")
print(len(spar), len(plate))            # 192 96
```

The body takes the elements 1 to 192 as its wetted surface, and the plate is added as a thin
sheet of the body ([Special surfaces](special-surfaces.md#2-thin-plates)). The spar is placed
20 m from the cylinder and held fixed for now (`modes=[]`, §5.3), since it has no mass
properties yet:

```python
s = m.bodies.add("spar", mesh="spar", ranges=[(1, 192)], position=[20.0, 0.0, 0.0], modes=[])
s.dipoles.add("plate", mesh="spar", ranges=[(193, 288)])
```

Counting elements by hand is error-prone for a real mesh. `mesh.elements()` gives the centroid,
normal, area and corners of every stored element, and `hb.ranges` turns a selection of elements
(a true or false value per element, or a list of element numbers) into ranges. The plate, for
example, consists of the elements whose centroids lie below the bottom of the spar:

```python
el = m.meshes["spar"].elements()
below = el.centroids[:, 2] < -4.25
print(hb.ranges(below))                 # [(193, 288)]
```

## 4. Checking the geometry

Mistakes in a mesh, such as normals pointing the wrong way or panels missing, are best found
before a solve. A mesh answers for its own geometry, in its own coordinates; a body answers for
its geometry as placed in the water, together with its mass properties. None of these queries
needs a solve. The queries of a body check the rules of that body alone: the values of the water,
the body's mesh and placement, and its surfaces. The queries of the model, such as `m.panels()`
and `m.mesh_quality()`, and `m.check()` apply every rule of the model:

```python
g = m.meshes["cylinder"].geometry()
print(g.volume, g.waterplane_area)       # m^3, m^2

h = m.bodies["cylinder"].hydrostatics()
print(h.center_of_buoyancy, h.metacentric_height)
print(h.stiffness[2, 2])                 # heave restoring, N/m

q = m.mesh_quality()
print(q.min_per_wavelength)              # the fewest panels per wavelength, per body
```

`geometry()` computes the volume three times, from $x$, $y$ and $z$ (`g.volumes`); three values
that differ markedly point to a mesh that does not close at the waterline. `mesh_quality()` flags
elements whose orientation disagrees with their neighbours, open edges below the waterline, and
panels too coarse for the shortest wave; the same findings appear as warnings when the model is
checked or solved. The panels of the meshes as the engine uses them, placed and reflected, are
`m.panels()`, and `m.tessellation()` gives quadrilaterals for drawing them, for example with
Matplotlib. Every query, its owner and the columns it returns are listed in
[reference/queries](../reference/queries.md).

## 5. Bodies

### 5.1 Axes

HydroBEM uses three sets of coordinates:

- the *global axes* have $z$ pointing up, the calm free surface at $z = 0$ and the sea bed at
  $z = -h$;
- the *body axes* are fixed to a body. The body item `position` places their origin in the global
  axes, and `yaw` turns them about the vertical, in degrees; the yaw is applied before the
  translation;
- the *mesh coordinates* are those the mesh is stored in. A body without trim uses its mesh as
  given, so its mesh coordinates are its body axes; a trimmed body places its mesh first (§7).

The points of a body, such as its centre of gravity, are given in body axes. Its first-order
results are in body axes too, with moments about its *rotation centre* $R$, the body item
`rotation_center` (the origin by default). The conventions are defined in
[theory §1.1](../theory/01-boundary-value-problem.md#11-coordinates-and-conventions).

### 5.2 Mass properties

The motions need the mass, the centre of gravity and the inertia of every free body. HydroBEM
assumes none of them: a body without a mass has none, and a solve of the motions then stops with
a message that names the missing item.

- `mass` is in kg, or `"displaced"` for the mass of the displaced water.
- `center_of_gravity` is the point $G$ in body axes.
- The inertia is given either as radii of gyration (`radii_of_gyration`, three values or a
  $3 \times 3$ matrix, in m) or as the six components `Ixx` to `Izz` of the inertia tensor in
  $\mathrm{kg\,m^2}$. The switch `inertia_input` says which, `"radii_of_gyration"` (the default)
  or `"components"`. Both are taken about `inertia_origin`, which is the centre of gravity by
  default.

```python
m.bodies["spar"].update(mass="displaced", center_of_gravity=[0.0, 0.0, -2.5],
                        inertia_input="components", Ixx=2.0e4, Iyy=2.0e4, Izz=1.5e3)
```

### 5.3 Modes, fixed bodies and moorings

A body moves in its six rigid modes: surge, sway, heave, roll, pitch and yaw. The item `modes`
lists the modes in which it is free; by default all six. A structure that is held fixed, but whose
wave forces are wanted, has `modes=[]`. It still affects the waves around the other bodies.

```python
m.bodies["spar"].modes = ["surge", "heave", "pitch"]
```

External springs and dampers, such as a mooring, are the $6 \times 6$ matrices
`additional_stiffness` and `additional_damping` in SI units, taken about the rotation centre
unless `stiffness_origin` or `damping_origin` names another point:

```python
k = np.zeros((6, 6))
k[0, 0] = k[1, 1] = 5.0e3                # N/m in surge and sway
m.bodies["spar"].additional_stiffness = k
```

Every item of a body is listed in
[reference/inputs-bodies §1.1](../reference/inputs-bodies.md#11-items); the equations of motion
are in [theory §3.1](../theory/03-motions-and-hydrostatics.md#31-rigid-body-motions).

## 6. Several bodies

Bodies are added one after the other, each with its own mesh (or the same mesh) and its own
position. They are solved together, so each feels the waves radiated and scattered by the others.
The spar above already stands 20 m from the cylinder. A connection between two bodies, such as a
hawser, is a *coupling*: a stiffness or damping between the forces on one body and the motions of
another.

```python
kc = np.zeros((6, 6))
kc[0, 0] = -1.0e3                        # the cylinder's surge force from the spar's surge motion
m.bodies["cylinder"].couplings.add(body="spar", stiffness=kc)
```

A coupling acts only on the rows of the body it belongs to; give the opposite direction as a
coupling of the other body. The rules are in
[reference/inputs-bodies §1.3](../reference/inputs-bodies.md#13-couplings-between-bodies).

## 7. A trimmed body

A hull floating at an angle, or at a draft other than that of its mesh, needs a new waterline.
Rather than meshing each condition, give the whole hull, above the water too, and set the body
item `trim` to `[elevation, pitch, roll]` (in m, degrees and degrees). When the problem is built,
the engine raises, pitches and rolls the hull and cuts it at $z = 0$; panels that cross the
waterline are cut, and patches are remapped. Every surface of the body is cut with it, and the
`ranges` keep counting the elements of the original mesh.

```python
hull = cylinder_panels(1.0, 1.0) + [0.0, 0.0, 0.5]     # 0.5 m above and below the water
m.meshes.add("hull_full", kind="panels", vertices=hull, symmetry="xz_and_yz")
t = m.bodies.add("trimmed", mesh="hull_full", trim=[-0.1, 2.0, 0.0], position=[0.0, 20.0, 0.0],
                 interior_surface="waterline")
t.trimmed_mesh("trimmed_wet").write("out/trimmed_wet.gdf", overwrite=True)
```

A trimmed body generates its interior surface from the new waterline (one of `"waterline"`,
`"graded"` and `"projected"`; [Special surfaces](special-surfaces.md#1-the-interior-surface)),
and it cannot have air cushions. A free-surface plate on it, such as the cover of a moonpool, is
generated from the new waterline as well: give a point inside the loop instead of a mesh
([Special surfaces](special-surfaces.md#5-plates-on-the-free-surface)). `trimmed_mesh` returns the
cut surface as a new mesh of the model, for inspection or for writing.

## 8. Leaving an object out

To compare a model with and without one of its parts, there is no need to delete the part: set its
item `enabled` to `False`. The solve then treats the model as if the object were absent, together
with everything that belongs to it, while the object and its items stay in the model and in the
model file. Bodies, tanks, thin plates, porous plates (of a body and fixed), air cushions,
free-surface plates, lids and couplings all have this item.

```python
for name in ("spar", "trimmed"):
    m.bodies[name].enabled = False
r = m.solve()                           # the cylinder alone
print([b.name for b in r.bodies])
m.bodies["spar"].enabled = True
```

The results number only the enabled bodies, and a coupling that names a disabled body is left out.
`m.check()` also checks each disabled object as if it were enabled, and warns when it would no
longer fit the model.
