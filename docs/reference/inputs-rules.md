# Inputs: model rules

## 1. Model rules (validation)

The type and range of every item, and the object a reference names (a mesh,
a body), are checked when it is set. The rules
between items and objects are applied by `m.check()` (which raises the first
broken rule as a `ValueError` and returns the warnings) and by every solve, in
a fixed order before any computation; each failure has a specific message
naming the object and the item. The rules, in the order applied:

1. Option combinations that would silently do nothing are rejected: field
   points without a field output (and a field output without points), the
   mean drift of a moving body without `first_order.motion`, `excitation_method = "haskind"`
   without radiation, a control surface without `second_order.drift_control_surface` (and that
   drift without a control surface on every body), `qtf_newman` without a drift switched on,
   `qtf_difference` and `qtf_sum` both off with a QTF switched on, a free-surface mesh
   (`free_surface_method = "mesh"`) without `qtf_potential_direct` or `qtf_potential_indirect`,
   `control_surface_waterline_term = "body"` without `second_order.drift_control_surface`. An
   item that the choice of a switch needs and that has no default must be given (the switches
   are described at the head of reference/inputs-model). With these come the rules of the
   second-order outputs:
   - the drifts and the QTFs need the diffraction;
   - a free-surface mesh consists of panels in $z = 0$;
   - the potential QTF is computed in the low order, in the open sea or by one wall (then on the
     generated panels), and without damping lids, fixed dampers, pressure surfaces, tanks, thin
     sheets or dipole dampers; a body with modes or generalized modes needs
     `first_order.motion`;
   - the pressure drift and the QTFs with generalized modes are computed in the low order,
     without pressure surfaces or tanks, and need a displacement that is continuous on the
     wetted surface, except for a vertical jump at the edge of a free-surface plate
     (reference/inputs-bodies §2.5).

   The free surface itself (its waterlines, inner radius and panels, reference/inputs-orders §3)
   is checked when its zones are made, by `m.check()`, the solve and the queries of the free
   surface.
2. Mesh validity (reference/inputs-meshes) and the items `mesh` and `ranges`
   (reference/inputs-meshes §3):
   - the ranges lie within their mesh, and an element has one role within a body ("element 12 of
     the mesh hull is taken by both the dipoles and the tanks (an element has one role in a
     body)");
   - every body has a wetted surface, a dipole or a free-surface plate (not dipole dampers alone),
     and a wetted surface, sheet or plate is not empty;
   - no wetted element lies wholly in the free surface $z = 0$: a waterplane cover belongs to the
     interior surface, which `"detect"` finds, or is a lid or a free-surface plate, or is left out;
     and no thin sheet lies wholly in $z = 0$;
   - the surfaces that take elements of the body's mesh (the dipole dampers, the tanks with their
     sheets, the pieces of the modes) name elements of that mesh;
   - a control surface holds its body alone, reaching no other body and holding no damping lid
     (reference/inputs-bodies §2.7);
   - with a trim (reference/inputs-bodies §1.2) the interior surface is generated, no pressure
     surface and no lid takes the body's elements, and a free-surface plate is only generated from
     the trimmed waterline (`inside`, a point in the water of a loop; no plate mesh, and not a body
     of plates only; reference/inputs-bodies §2.4); a generated plate naming another body (`body`)
     stands only on a body without trim, and that body is trimmed;
   - a lid `placed_with` a body takes elements of that body's mesh;
   - `body_points` need a wetted surface or a free-surface plate (a body of sheets only has none).
3. Lids at $z = 0$ with vertical normals; pressure surfaces horizontal, $z \le 0$; interior surfaces and
   free-surface plates at $z = 0$ (elements of either orientation turned to the convention: downward
   for pressure surfaces and plates, upward for interior surfaces and lids; every element, patches
   of every kind among them, reference/inputs-bodies §2.4). A free-surface plate overlaps no other surface of $z = 0$:
   another plate (of its body or of another), a lid, an interior surface (given or generated) or a
   pressure surface in $z = 0$ (a point of $z = 0$ has one condition); an overlap is measured on the
   elements as horizontal polygons and refused when its width (area over diameter) exceeds a tenth
   of the smaller element's diameter, so that edges touching and the slivers of two meshes that
   approximate one curve by different chords pass. Lids may overlap each other (their columns
   add). A thin sheet (a dipole, a dipole damper, a fixed damper) whose edge lies in $z = 0$ meets
   a free-surface plate or a lid only along the edges of their elements (within 1 % of an element's
   diameter): the potential of the plate or lid jumps across the sheet, which an element cannot
   hold inside it (theory §9.7). The plates of a tank's free surface lie at the tank's surface
   level, and the rules of this item apply to them in the tank's free surface (in the tank's
   problem: no overlap with each other, the sheets inside the tank meeting them on their edges).
4. Something to solve: radiation (with modes) or diffraction; with both off
   the model is hydrostatics only (route `hydrostatics`: the `hydrostatics_*` quantities
   and nothing per frequency; `environment.omegas` may then be empty).
5. `motion=True` needs `radiation=True`, `diffraction=True`, ordinary frequencies,
   centre of gravity and mass of every moving body (its inertia the side `inertia_input`
   chooses, zero if not given).
6. Frequency limits need radiation and non-empty modes; diffraction and field
   points need ordinary frequencies; headings need diffraction.
7. Environment ranges; domain/width consistency; channel headings $0^\circ$ or $180^\circ$;
   bodies strictly on the fluid side of walls; the positions of the field points in the fluid
   domain, each in the sea or in its tank (reference/inputs-model-objects §3).
8. Interior surfaces, lids, pressure surfaces in $z = 0$ and free-surface plates require
   logarithmic integration; in the higher
   order lids require the total diffraction formulation, in the low order thin
   sheets (dipoles and dampers) the scattered one when diffraction is solved.
9. Finite-depth (and channel) zero-frequency modes with net volume flux are
   divergent and are rejected when requested (no finite gauge mass).
10. With `motion=True`, the inertia of the free coordinates of every body (the rigid inertia on
    its free rigid modes and `mode_mass` on its generalized coordinates) is positive
    semi-definite, and each massless direction of it moves water (has a normal velocity on the
    wetted surface, the thin sheets, the free-surface plates or the tank walls); the refusal names
    the coordinates (theory §3.1).

A valid model must then match one of the implemented solver routes (the
supported combinations, §2); otherwise it is
rejected with `UnsupportedProblemError` (Rust: `ErrorKind::Unsupported`) naming the
reason, by `m.check()`, `m.route()` and the solve alike (the same error and message).
`m.route()` applies the rules and returns the name of the route without solving.

## 2. Supported combinations

Every feature works in infinite and finite depth unless stated, for one or several bodies, with
symmetry sectors and with the limits $\omega = 0$ and $\omega = \infty$ of the added mass. The combinations of a
feature with a method, a domain or an output (checked by `m.route()` against the engine) are
those of Table 1.

*Table 1. The supported combinations of the features with the two methods.*

| Feature | Low order | Higher order |
|---|---|---|
| Open sea, walls (`x_wall`, `y_wall`, `orthogonal_walls`) | yes | yes |
| Wavemakers in the walls (a body in a wall plane, radiation only) | yes | yes |
| Channel (finite depth) | yes | yes |
| Trim | yes; tanks cut at their own free surface | yes; the interior surface by the method `"waterline"` |
| Interior free surface (irregular frequencies) | yes | elements of the body mesh or a patch mesh of its own, or `"waterline"` on a trimmed body |
| Damping lids | yes; no momentum drift | placed with a body (elements of its mesh) or a patch mesh in global coordinates; no momentum drift |
| Thin sheets (dipoles) | yes; no pressure drift, control-surface drift, field velocity or body velocity (the source formulation of the velocities) | planar dipole patches of the body mesh or of a patch mesh of their own; no pressure drift |
| Generalized modes | yes; the pressure drift and the quadratic and potential QTFs with their generalized forces (not with pressure surfaces or tanks; the displacement continuous on the wetted surface) | yes; no pressure drift |
| Internal tanks | yes, with field points inside and the pressure drift (the tank fluid on its walls; not with plates of the tank's free surface); plates of the tank's free surface (inputs-bodies §2.2), not on a trimmed body | yes, with field points, thin sheets and plates of its free surface; no pressure drift |
| Dipole dampers of a body, fixed dampers | as thin sheets; no limits $\omega = 0, \infty$ and no momentum drift; no Haskind force with a fixed damper | as thin sheets; fixed dampers only on the mesh of a body at the origin without yaw or trim |
| Prescribed-pressure surfaces | yes; the limit $\omega = \infty$ only for surfaces in $z = 0$; not on a trimmed body | elements of the body mesh or a patch mesh of their own; likewise $\omega = \infty$ |
| Free-surface plates | yes, both formulations, $\omega = 0$ and $\omega = \infty$, field and body velocity; open sea, walls and a channel; bodies with tanks; the momentum drift and Newman's approximation of it; the pressure drift, its quadratic QTFs and Newman's approximation of it when every plate edge lies on a wall (of its body or another, within a tenth of the plate edge's length; a wall of the domain) or between plate elements of its body (an edge against another body's plate is bordered by water whose plates move apart: free), not with pressure surfaces (theory §11.2); the potential QTF, both methods, for the same plates, the free surface the generated panels or a given mesh that ends at the plates (theory §12.7, §12.8); the control-surface drift, its quadratic QTFs and Newman's approximation of it for the plates of the body whose control surface it is, under the same rule of the edges (theory §11.3; a plate body inside another body's control surface is a second body there, refused); not a plate edge bordered by free surface in the pressure or control-surface drift or the potential QTF, $\omega = \infty$ with a thin sheet hanging from a plate, the cells of the potential QTF's free surface, wavemakers or a plate mesh on a trimmed body (the plate of a trimmed hull is generated) | elements of the body mesh or a patch mesh of their own; likewise (the pressure drift not with generalized modes) |
| Momentum drift | open sea, no lids; with free-surface plates | open sea, no lids; with free-surface plates |
| Pressure drift | open sea, walls or a channel, except with the features above that exclude it | likewise |
| Control-surface drift | yes (given, cylinder, ellipsoid or automatic control surfaces); not with thin sheets, generalized modes, dipole dampers (of a body) or fixed dampers, tanks or pressure surfaces | yes; not with generalized modes, dipole dampers (of a body) or fixed dampers, tanks or pressure surfaces |
| Field and body velocity | open sea, walls, interior surfaces, lids, pressure surfaces and a channel; not with thin sheets | yes |
| Quadratic QTFs | where the drift of their method is available (pressure integration, control surface), with its exclusions | likewise; the pressure integration not with generalized modes |
| Potential QTF (second-order potential) | open sea, `x_wall`, `y_wall` (with a wall the generated panels); fixed or floating bodies, also with generalized modes and interior surfaces; with free-surface plates whose edges lie on walls; not with `orthogonal_walls`, a channel, damping lids, fixed dampers, pressure surfaces, tanks, thin sheets or dipole dampers of a body | no |
| Newman's approximation | where the drift it is made from is available | likewise |

Interior surfaces, thin sheets, pressure surfaces, free-surface plates and lids combine with each
other, in both methods. Higher-order lids need the total diffraction
formulation, low-order thin sheets the scattered one (both chosen by `"auto"`; rule 8). The
higher order takes at most 32 coordinates (6 per body, the generalized and the pressure modes).
