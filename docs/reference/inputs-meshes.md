# Inputs: meshes

## 1. Low-order mesh (kind `panels`)

- A mesh is an object of the list `m.meshes`, named; bodies and surfaces refer to it by its name,
  the item `mesh`, with the item `ranges` for a part of its elements (§3). A mesh that another
  object refers to cannot be removed.
- Elements (panels): the item `vertices`, `P × 4 × 3` vertex coordinates (mesh coordinates, metres) in raw order.
  The vertex order defines the outward normal (right-hand rule, pointing into
  the fluid). A triangle is given by repeating a vertex (two consecutive
  coincident vertices). **[B]**
- `reference_length` > 0 (the length scale of the normalized files), `file_gravity` (the
  gravity written in the file), `label` (the title), the symmetry planes `symmetry`:
  `"none"`, `"yz"` (the stored elements reflected about the plane $x = 0$, $x \to -x$; the GDF
  field ISX), `"xz"` (about the plane $y = 0$, $y \to -y$; ISY) or `"xz_and_yz"` (both: one quadrant
  stored, four copies).
- Panel geometry rule: every quadrilateral is replaced by its orthogonal
  projection onto the plane through the midpoints of its four sides (normal
  $(v_2 - v_0) \times (v_3 - v_1)$, through the vertex mean); raw vertices and their order are
  kept; the projected panel must then be a valid, strictly convex planar polygon. There is
  no switch to bypass the projection. **[C]** (theory §14)
- Validation (each failure is a `ValueError` naming the element, one-based):
  fewer than three distinct vertices, zero area, a non-convex or
  self-intersecting projected polygon, non-finite coordinates. Vertices coincide
  within 1e-6 of the panel diameter, a polygon is strictly convex when the sine of every
  turning angle exceeds 1e-6, the area is zero below 1e-6 of the squared diameter
  **[C]** (theory §14).
- Readers: `m.meshes.load(name, gdf, *, zero_trailer=False)` reads a low-order GDF file
  **[B]**: line 1 a title, line 2 the reference length and the gravity, line 3 the two
  symmetry flags (0 or 1 for the planes $x = 0$ and $y = 0$), line 4 the panel count, then four
  vertices (twelve numbers) per panel, free format (text after the numbers of a line is a
  comment, as `end of panel 1`; a line that begins with text is refused, also in a low-order
  CSF file); a trailing record of zeros is accepted
  only with `zero_trailer=True`. The file's gravity is kept as the item `file_gravity` (the
  solve uses `environment.gravity`), and the absolute path of the file as `source_file`
  (with `source_spl` for the SPL file of a patch mesh; `""` for a mesh given as arrays),
  kept with the model for its clients and not used by the solve.
  `m.meshes.add(name, kind="panels", vertices=v, reference_length=1.0,
  symmetry="none", label="")` makes a mesh from arrays; `mesh.write(gdf, overwrite=False)`
  writes it as a GDF file, every number at full precision.

## 2. Higher-order mesh (kinds `flat_patches`, `spline_patches`, `analytic_patches`)

- Kind `flat_patches`: each patch is the bilinear map of its four vertices (the item `vertices`,
  `P × 4 × 3`), which must be distinct, coplanar (1e-6 of the diameter) and strictly convex
  (no projection; no triangles).
- Kind `spline_patches`: each patch is a tensor-product B-spline surface with geometry orders 2..8 in u
  and v, complete knot vectors and vector control coefficients; with the item `weights` a
  rational one (NURBS, Piegl & Tiller 1997): $x = \sum w N_i N_j P / \sum w N_i N_j$ with positive
  weights of the shape $(M_v, M_u)$, which represents the circle, the cylinder and the sphere
  exactly (the quarter circle of order 3 with the middle weight $\cos 45^\circ$); without `weights`
  (or all weights 1) the non-rational patch. Knot vectors are mapped affinely to $[-1, 1]^2$. The
  parametric cross product $x_u \times x_v$ points into the body **[B]**. Patches must be C1 inside;
  a C0 knot is rejected. From arrays: `m.meshes.add(name, kind="spline_patches", discretization=d,
  patches=[{"type": "SplinePatch", "u_order": ..., "v_order": ..., "u_knots": ..., "v_knots":
  ..., "coefficients": ..., "weights": ...}, ...])`, the control coefficients of shape
  $(M_v, M_u, 3)$, $M$ = spans + order $- 1$. The GDF format has no weights: a rational patch is given from
  arrays (or kept in a model file), and `mesh.write` refuses a mesh with weights other than 1.
- Potential discretization per patch (the item `discretization`, one row
  `[nu, nv, ku, kv, iquo, iqui]` per patch): the B-spline spans (nu, nv) $\ge 1$ and orders
  (ku, kv) in 2..6 of the potential on $[-1, 1]^2$, and the Gauss orders of the outer (test)
  and inner (source) integrals (iquo, iqui). These come from an SPL file, are given directly
  (`discretization=`), or follow from a panel size (`panel_size=` $s$, m): $\mathrm{nu} = \lfloor L_u/s + 10^{-5} \rfloor + 1$
  and nv likewise, with $L_u$, $L_v$ the longer of the two patch edges in that direction (arc
  lengths for B-spline and analytic patches), orders 3 and Gauss orders 3 (outer) and 4
  (inner) unless `spline_order`, `outer_quadrature`, `inner_quadrature` are given; more than
  $10^6$ spans along an edge is an error. The integration of the near-singular and self-patch
  terms is fixed by the method (theory §7).
- Readers: `m.meshes.load(name, path, *, higher_order=False, spl=None,
  discretization=None, panel_size=None, spline_order=None, outer_quadrature=None,
  inner_quadrature=None, zero_trailer=False)` (a patch mesh with `higher_order=True`); from
  arrays `m.meshes.add(name, kind="flat_patches", vertices=v, discretization=d,
  reference_length=1.0, symmetry="none")`. The reader accepts both higher-order GDF
  representations and the analytic shapes below; `mesh.write(gdf, spl=...)` writes a flat or
  B-spline mesh. **[B]** A higher-order GDF file has the first three lines of the low-order
  file, then line 4 the patch count and the geometry code (0 flat, 1 B-spline, negative an
  analytic shape). With code 0 each patch is twelve free-format vertex coordinates. With
  code 1 each patch holds its span counts $(n_u, n_v)$, its geometry orders $(k_u, k_v)$, a
  u-knot vector of length $n_u + 2k_u - 1$, a v-knot vector of length $n_v + 2k_v - 1$, and
  $(n_u + k_u - 1)(n_v + k_v - 1)$ vector control coefficients, the u index varying fastest.
- An SPL file **[B]** has one header line followed, for every patch in GDF order, by four
  pairs, one per line: the spans (nu, nv), the orders (ku, kv), the outer Gauss orders and
  the inner Gauss orders (each for u and v). The spans are always present. A pair among the
  other three may be omitted only when the corresponding `spline_order`,
  `outer_quadrature` or `inner_quadrature` keyword supplies one global value for both
  directions; a value is never supplied both ways. The two directional quadrature orders
  must be equal. No other configuration file is read. When `spl` is None, `discretization`
  is required: one row for all patches, or one row per patch. **[H]**
- The `ranges` of a surface (§3) count patches.
- Analytic patches (kind `analytic_patches`, the list `patches` of `{"type": "AnalyticPatch",
  "map": ..., ...}`): exact maps of five kinds, a revolved segment (`"line"`), a revolved
  elliptic arc (`"arc"`), a bilinear quadrilateral, the surface between two ellipses
  (`"ellipses"`) and between a segment and an elliptic arc (`"segment_arc"`), with the
  profile `"linear"`, `"smooth"` ($3t^2 - 2t^3$), `"quadratic_start"` ($t^2$) or `"quadratic_end"`
  ($2t - t^2$) in $t = (v + 1)/2$; the items `a`, `b`, `theta0`, `theta1`, `p` and `vertices`
  are the parameters of the kind.
- Named analytic shapes: `m.meshes.analytic(name, shape, discretization, parts=..., mapping=...,
  symmetry="xz_and_yz", **dimensions)` (one discretization row for all patches, or one per patch) makes the quadrant
  $x, y \ge 0$ of a shape as analytic patches (Table 1).

  *Table 1. The named analytic shapes, their dimensions and parts.*

  | `shape` | dimensions (m) | parts (default: all but the last) |
  |---|---|---|
  | `"circular_cylinder"` | `radius`, `draft` | `side`, `bottom`, `waterplane` |
  | `"elliptic_cylinder"` | `a`, `b`, `draft` | `side`, `bottom`, `waterplane` |
  | `"sphere"` | `radius` | `hull`, `waterplane` |
  | `"ellipsoid"` | `a`, `b`, `c` | `hull`, `waterplane` |
  | `"barge"` | `half_length`, `half_beam`, `draft` | `end`, `side`, `bottom`, `waterplane` |
  | `"moonpool_cylinder"` | `radius`, `draft`, `moonpool_radius` | `side`, `bottom`, `moonpool_wall`, `moonpool_surface` |
  | `"cylinder_with_dipole_discs"` | `radius`, `draft`, optional `moonpool_radius`; `discs`: per annular dipole disc its outer radius, inner radius and depth | fixed: the cylinder's side and bottom (and moonpool wall), then the discs |

  The patches follow the order of `parts`. The `waterplane` is the interior free surface, the
  `moonpool_surface` the free surface inside the moonpool (a lid). `mapping="nonuniform"` (the
  cylinders, the sphere and the moonpool cylinder) takes the profiles with zero slope at the
  patch ends, which cluster the panels there; the pressure drift is not available when such a
  profile meets the waterline. `m.meshes.load(..., higher_order=True)` reads the analytic codes
  of the higher-order GDF format **[B]** into these shapes (the file's geometry code IGDEF $-1$, $-2$,
  $-3$, $-4$, $-5$, $-7$, $-34$ in the order of Table 1; its patch count NPATCH the first parts, for $-5$
  with NPATCH = 1 the bottom; the mapping flag INONUMAP of its data lines the mapping). The
  `moonpool_radius` of the moonpool cylinder, and of the cylinder with dipole discs when it is
  not 0, must be smaller than `radius`. A dipole disc not below the bottom (depth $\le$ `draft`) must
  lie in the water: its inner radius at least `radius`, or its outer radius at most
  `moonpool_radius`.

## 3. The items `mesh` and `ranges`

Every surface of the model names its mesh by the item `mesh`, the name of a mesh of `m.meshes`,
and its elements by the item `ranges`: one-based, inclusive ranges `[(first, last), ...]` of the
stored elements of that mesh, counted before the symmetry expansion (the elements of a low-order
mesh are its panels, those of a patch mesh its patches); `ranges` unset (None) is every stored
element:

```python
m.bodies.add("spar", mesh="spar")                                 # every element of the mesh "spar"
m.bodies.add("spar", mesh="spar", ranges=[(1, 96)])               # the elements 1..96
m.bodies.add("spar", mesh="spar", ranges=[(1, 96), (129, 140)])   # a union of ranges
```

- `ranges` is an integer array (N, 2), kept sorted by the first element. Overlapping ranges, a
  range with first < 1 or last < first and an empty list of ranges are refused when the item is
  set; a range beyond the elements of its mesh when the model is checked.
- The surfaces with these two items: a body (its wetted surface), its dipoles, dipole dampers,
  tanks and the sheets of a tank, pressure surfaces, free-surface plates, the lids and the fixed
  dampers. The interior
  surface, the control surface and the free surface of the potential QTF given as meshes have
  them as prefixed items (`interior_surface_mesh`, `interior_surface_ranges`,
  `control_surface_mesh`, `control_surface_ranges`, `second_order.free_surface_mesh`,
  `free_surface_ranges`). A piece of a generalized mode has `ranges` only, of the body's mesh
  (reference/inputs-bodies §2.5).
- The ranges always count the elements of the mesh as stored, also when the body is trimmed
  (reference/inputs-bodies §1.2): the engine maps them onto the elements it cuts.
- The same mesh may be named by any number of bodies and surfaces. Within one body an
  element of a mesh has at most one role: the wetted surface (the body's `mesh`), a dipole, a
  dipole damper, a tank's walls, a sheet of a tank, a pressure surface, a free-surface plate, the
  interior surface, a lid placed with the body or a fixed damper on the body's mesh (reference/inputs-model-objects; the roles: reference/queries §1). The elements of the body's
  mesh that no role takes are not used. A GDF file that holds the hull, its sheets and its tank walls is one
  mesh, and each surface names its elements:

  ```python
  m.meshes.load("fpso", "fpso.gdf")
  m.bodies.add("fpso", mesh="fpso", ranges=[(1, 400)])
  m.bodies["fpso"].tanks.add(mesh="fpso", ranges=[(401, 460)], density_ratio=0.85)
  ```
- `hb.ranges(selection)` gives the ranges of a selection of elements: a boolean mask over the
  stored elements of a mesh (`mask[k]` is element k + 1) or one-based element numbers in any order;
  nothing selected is refused (`ranges` None would be the whole mesh). With the
  elements of the mesh (§4) a selection by geometry is a mask:

  ```python
  el = m.meshes["spar"].elements()
  lid = (el.normals[:, 2] > 0.5) & (abs(el.vertices[..., 2]).max(axis=1) < 1e-6)  # the waterplane
  m.bodies["spar"].update(ranges=hb.ranges(~lid), interior_surface="mesh",
                          interior_surface_mesh="spar", interior_surface_ranges=hb.ranges(lid))
  ```

  (The interior surface's default `"detect"` takes these elements by itself, reference/inputs-bodies §2.6.)

## 4. The queries of a mesh

A mesh answers for its own geometry, in mesh coordinates: no density, gravity, placement or trim
(those are the body's, reference/inputs-bodies §1 and reference/queries).

- `mesh.geometry()` (`hb.MeshGeometry`): the volume closed by the mesh and the plane $z = 0$ (an
  open wetted surface closed by its waterplane, a closed hull its whole volume): its three
  estimates by the divergence theorem, the integrals of $x n_x$, $y n_y$ and $z n_z$ (`volumes`), and
  their mean (`volume`); the centre of volume (its first moments over the z estimate; 0 when that
  estimate is 0); the
  surface `area`; the `bounds` (the smallest and largest coordinates of the vertices, of a patch
  a $17 \times 17$ grid of its points); the waterplane at $z = 0$: `waterplane_area` (the integral of
  $-n_z$), `waterplane_centroid` (0 without a waterplane) and `waterplane_moments`, the integrals of
  $x^2$, $y^2$ and $x y$ about
  the origin. Over the whole mesh, its reflections included, with the quadrature of the
  hydrostatics (theory §3.2).
- `mesh.elements()` (`hb.MeshElements`): per stored element, without the reflections, its
  one-based number (`element`, the numbering of `ranges`), `centroids` [n, 3], unit `normals`
  [n, 3] into the fluid, `areas` [n], `vertices` [n, 4, 3] and `distinct_vertices` [n] (3 or 4),
  mesh coordinates (m, $\mathrm{m}^2$); a patch gives its corners and the point and normal at the
  centre of its parameters.
- `mesh.tessellation(divisions=8, include_reflections=True)` (`hb.Tessellation`): a
  quadrilateral mesh of the mesh for display, mesh coordinates (`divisions` quads per patch in
  each direction, a pair `(du, dv)` or `"spans"`, one per low-order panel; role `""`, body, tank
  and owner $-1$, part 0; reference/queries §2); `mesh.patch_points(element, uv)`
  (`hb.PatchPoints`): the `points` and `normals` of a patch at parameters in $[-1, 1]^2$.
- `mesh.quality()`: the geometric findings of `m.mesh_quality()` (reference/queries §4) for the
  mesh alone: per element the panel or span size, the aspect and the flags of orientation, free
  and non-manifold edges and the open ends of the waterline (no wavelength: the panels or spans
  per wavelength infinite, no `"coarse"`); the three volumes and the edge counts (of a patch mesh,
  its patch sides; theory §14) of the surface of all its elements closed by the waterline $z = 0$.
- `mesh.write(gdf, spl=None)` writes a `panels`, `flat_patches` or `spline_patches` mesh as GDF
  (and SPL; §1–§2; an `analytic_patches` mesh is kept in the model file, reference/inputs-model §4); a path ending in
  `.hbg` writes the geometry file of a mesh of any kind except `trimmed_patches` (§5).

A mesh of kind `trimmed_patches` is made by the trim of a body (`body.trimmed_mesh`, reference/inputs-bodies §1.2): its items
name its `source` mesh and hold the `trim` (elevation, pitch, roll), the `trim_elements` (per
element the one-based source patch, its reflection and cut), their `parameter_ranges` (per
element the part u0, u1, v0, v1 of the source patch it covers),
`discretization` and
`source_elements` (the one-based source element of every element; with a generated interior surface
the patches of the mesh and those of the interior surface are a new source mesh
`<name>_source`).

## 5. The geometry file (.hbg)

A geometry file holds one mesh of any kind except `trimmed_patches` (low-order panels, or flat,
B-spline (also rational) or analytic patches) with its discretization: the role of a GDF file with its SPL
file, also for the meshes a GDF file cannot hold (rational and analytic patches, patch names).
Its text is JSON (RFC 8259, UTF-8): one object whose members are those of Table 2.

*Table 2. The members of the geometry file.*

| member | |
|---|---|
| `"format"` | `"hbg"` (required) |
| `"version"` | `3` (required; a file of another version is refused) |
| `"kind"` | `"panels"`, `"flat_patches"`, `"spline_patches"` or `"analytic_patches"` (required) |
| the items of the mesh | `label`, `reference_length`, `symmetry`, `vertices` (panels and flat patches), with their names, kinds and units (metres) of §1–§2; unset items take their defaults |
| `"patches"` | spline and analytic patches: an array with one object per patch, its members the items of a `SplinePatch` (`u_order`, `v_order`, `u_knots`, `v_knots`, `coefficients`, `weights`) or an `AnalyticPatch` (`map`, `profile`, `a`, `b`, `theta0`, `theta1`, `p`, `vertices`), and optionally `"name"` (unique in the mesh, not only digits), `"discretization"` (its row) and `"type"` (the patch type of the kind) |
| `"discretization"` | patch kinds: one row `[nu, nv, ku, kv, iquo, iqui]` for every patch, or one row per patch |
| `"panel_size"` | patch kinds: the discretization from a panel size (§2, the arc lengths of the patch edges measured on the patches), with the optional `"spline_order"` (2..6), `"outer_quadrature"`, `"inner_quadrature"` (1..64) |
| `"info"` | any value, at the top and in a patch: for the writer of the file; not read and not kept |

Arrays are nested lists in the shape of the item (`coefficients` $M_v \times M_u \times 3$, `weights`
$M_v \times M_u$, `vertices` $P \times 4 \times 3$). A member that is no item of the mesh or patch type, an item of
another mesh kind, an item the engine sets (`source_file`, `source_spl`, `file_gravity`, the
items of a trim), a value of the wrong kind and a JSON syntax error (with its line and column)
are refused. The discretization of a patch mesh is given once: per patch, by the member
`discretization`, by `panel_size`, or, for a file without one, by the reader (`discretization=`,
`panel_size=`); a `panels` mesh has none.

- `m.meshes.load(name, path)` reads a file whose name ends in `.hbg` (any case; no `spl` or
  `zero_trailer`); the mesh keeps the path (`source_file`).
- `m.meshes.from_hbg(name, data)` makes the mesh from the object of a file (a dict as
  `json.load` gives it, NumPy arrays taken as lists, or the JSON text), as a program passes it
  without a file; `mesh.to_hbg()` gives the object of a mesh, and `mesh.write("x.hbg")` writes
  it: the items set, in the order of the schema, every number at full precision (written and read
  back, a mesh is the same item by item).

An example, the side of a floating cylinder (quadrant) as an exact rational patch:

```json
{
  "format": "hbg",
  "version": 3,
  "kind": "spline_patches",
  "symmetry": "xz_and_yz",
  "patches": [
    {
      "name": "side",
      "u_order": 3,
      "v_order": 2,
      "u_knots": [0, 0, 0, 1, 1, 1],
      "v_knots": [0, 0, 1, 1],
      "coefficients": [[[1, 0, 0], [1, 1, 0], [0, 1, 0]], [[1, 0, -0.5], [1, 1, -0.5], [0, 1, -0.5]]],
      "weights": [[1, 0.7071067811865476, 1], [1, 0.7071067811865476, 1]],
      "discretization": [4, 2, 3, 3, 3, 4]
    }
  ]
}
```
