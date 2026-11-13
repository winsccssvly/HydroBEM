# Working with the model

A problem in HydroBEM is one object, the model (`hb.Model`). Everything about the problem lives in
it: the meshes, the bodies, the water and the waves, what to compute and how. Once you know how the
model is organized and how its inputs are set, every other page of the guide is a matter of which
inputs to set. This page explains that organization, how to change and check a model, and how to
keep it in files.

The examples continue the cylinder of [Getting started](getting-started.md#2-a-first-problem-a-floating-cylinder):
a model `m` with the mesh `"cylinder"` and the body `"cylinder"`.

## 1. Objects and items

The inputs are grouped into *objects*, and each input value is a named *item* of one object. The
water depth, for example, is the item `depth` of the object `m.environment`, and the mass of a body
is the item `mass` of that body. Some objects exist once in every model:

- `m.environment`: the water (depth, density, gravity, walls) and the waves (frequencies and
  headings);
- `m.options`: how the problem is solved (the method, the number of threads);
- `m.first_order`: what the first-order solve computes (radiation, diffraction, motions, pressures);
- `m.second_order`: the mean drift forces and the second-order loads.

Other objects come in *lists*, and a model may have any number of them: the meshes `m.meshes`, the
bodies `m.bodies`, the damping lids `m.lids` and the fixed porous plates `m.fixed_dampers`.

An object that moves with another, or has no meaning without it, belongs to that object as a
*child*. The thin plates, tanks, air cushions and generalized modes of a body are lists of the body
(`hull.dipoles`, `hull.tanks`, ...), and a baffle inside a tank is a child of the tank. The whole
model therefore forms a tree:

```
Model
├─ meshes        the meshes, by name
├─ bodies        each with its mesh, placement, mass properties and the surfaces that move
│                with it: dipoles, dipole_dampers, tanks, pressure_surfaces,
│                free_surface_plates, generalized_modes, couplings
├─ lids          damping lids (at rest)
├─ fixed_dampers porous plates fixed in space
├─ environment   depth, gravity, density, walls, frequencies, headings
├─ options       method, threads, memory check
├─ first_order   radiation, diffraction, motions, field and body outputs, field points
├─ second_order  mean drifts, QTFs and the free surface of the second-order potential
└─ tool          text kept for other programs
```

The complete tree with every item, its type and its default is in
[reference/inputs-model](../reference/inputs-model.md).

## 2. Setting and reading items

An item is read and set like an attribute. Several items of one object are set at once with
`update`, and `reset` returns an item to its default:

```python
m.environment.depth = 50.0                               # one item
m.environment.update(density=1025.0, gravity=9.81)       # several at once
m.environment.reset("depth")                             # back to the default (deep water)
print(m.environment.depth)
```

Each value is checked as soon as it is set. A value of the wrong type raises `TypeError`, a value
out of range `ValueError`, and the message names the object and the item, for example
`bodies/cylinder: mass must be a finite number >= 0, got -1.0`. `update` is all or nothing: if
one of its values is refused, none of the items changes.

To see an object, print it. The printout lists the items that have been given; §2.1 explains the
line `unused` that some objects add. `items()` returns every item that applies to the object
(of a mesh, those of its kind), without its name, with its value as a dictionary, and `info` describes one item, its unit, default and
meaning:

```python
cyl = m.bodies["cylinder"]
print(cyl)
print(cyl.info("mass"))
values = cyl.items()
```

The same descriptions appear as help text in a Python editor or in Jupyter (`help(m.second_order)`),
and editors offer the item names as completions.

`items()` leaves out the object's name and its lists, so it can be used to copy a body:
`m.bodies.add("cylinder2", **cyl.items())` makes a second body with the same items.

### 2.1 Either-or inputs

Some inputs can be given in one of two or more ways. The frequencies, for instance, are either
radian frequencies (`omegas`) or periods (`periods`), and the inertia of a body is either its
radii of gyration or the six components of its inertia tensor. In each such case a *switch* item
says which way is used, and only that way is read:

```python
m.environment.update(frequency_input="periods", periods=[2.0, 4.0, 8.0])
m.environment.omegas = [1.0, 2.0]        # kept, but not used while frequency_input is "periods"
print(m.environment)                     # the last line names the unused items
```

Setting one way never clears the other, so you can switch back and forth without retyping. The
switches are `environment.frequency_input`, a body's `inertia_input`, `interior_surface` and
`control_surface` (with `control_surface_outer`), and `second_order.free_surface_method`. When a
switch chooses a way that needs an item without a default, such as the radius of a cylindrical
control surface, a missing value is reported by the check before the solve (§4).

## 3. Lists of objects

The objects of a list are made by `add` (at the end) or `insert` (at a position), and found by
name or by position:

```python
m.meshes.add("cylinder_fine", kind="panels", vertices=cylinder_panels(1.0, 0.5, 32, 12, 12),
             symmetry="xz_and_yz")
fine = m.bodies.add("fine", mesh="cylinder_fine", position=[10.0, 0.0, 0.0])
print(m.bodies.names(), m.bodies[1].name)
m.bodies.rename("fine", "buoy")
m.bodies.move("buoy", 0)                 # the buoy first
del m.bodies["buoy"]
del m.meshes["cylinder_fine"]
```

The order of a list is the order of its objects in the results and the output files. References
between objects use names, never positions, so reordering a list changes nothing else, and a
rename updates every reference to the object. An object that another one refers to cannot be
removed: deleting the mesh of a body raises a `ValueError`,
`meshes/cylinder: is used by bodies/cylinder: change or remove that reference first`, until the
body names another mesh or is removed itself. `clear()` empties a list.

A list itself is never assigned: `m.bodies = [...]` raises `TypeError`. Children are added to their
parent's list once the parent exists, for example `m.bodies["hull"].tanks.add(...)`.

The thin plates, porous plates, tanks, air cushions, free-surface plates and couplings of a body,
and the fixed porous plates of the model, may have names, unique within their
list (`hull.dipoles.add("plate", mesh=...)`), or none (`hull.dipoles.add(mesh=...)`). An object
with a name can be found by it, and messages name it.

## 4. Checking a model

The rules that relate several items or objects, such as "the motions need a centre of gravity",
are checked when the model is solved. They can also be checked without solving:

```python
warnings = m.check()        # raises at the first broken rule or unsupported combination
print(m.route())            # the name of the solver that will take the model
```

`m.check()` raises `ValueError` with a message that names the object and the rule, and otherwise
returns the warnings, which are also kept in `m.warnings`. A model can obey every rule and still
ask for a combination of features that the engine does not solve, for example the mean drift by
momentum conservation in a channel. For such a model `m.check()` raises
`hb.UnsupportedProblemError`, which names the combination, with the same message as `m.route()`
and the solve: a model that passes `m.check()` has a solver route, so one call before a long
solve is enough. Which features combine is shown in
[reference/inputs-rules §2](../reference/inputs-rules.md#2-supported-combinations).

## 5. The model and its results

`m.solve()` returns the results and also keeps them in the model, as `m.results`. Any change of an
input afterwards discards the kept results, because they no longer belong to the model:

```python
results = m.solve()
print(m.state)              # "calculated"
m.environment.headings = [0.0]
print(m.state)              # "reset": m.results now raises RuntimeError
```

A results object, once returned, stays valid: it holds its own copy of the values and of the
model that was solved (`results.model`). A parameter study is therefore a loop that changes an
input and solves again:

```python
for depth in (5.0, 10.0):
    m.environment.depth = depth
    r = m.solve()
    print(depth, r.added_mass[0, 2, 2])
m.environment.reset("depth")
```

`m.reset()` discards the results and keeps the inputs, `m.clear()` empties the model, and
`m.copy()` returns a copy of the inputs without the results. `m.close()`, or the end of a
`with hb.Model() as m:` block, frees the memory of the model at once.

The Python objects hold no copy of the inputs: reading an item reads it from the engine, and
setting an item sets it there. The same model can be built and solved from a Rust program
([reference/rust-api](../reference/rust-api.md)) or a C program
([reference/c-api](../reference/c-api.md)).

## 6. Model files and results files

A model is saved as a *model file*, `.hbm`, a single file with every object and item of the model,
the meshes included. A mesh read from a GDF file is stored in it, so the GDF file is not needed
again, and every number keeps all its digits, so a saved and reloaded model gives identical
results. The results are saved as a *results file*, `.hbr`, which holds the model that was solved
as well:

```python
m.save("out/case.hbm", overwrite=True)
m2 = hb.Model.load("out/case.hbm")           # also reads the inputs of a results file
results = m2.solve()
results.save("out/case.hbr", overwrite=True)  # or m2.save_results(...)
again = hb.load_results("out/case.hbr")
print(again.model.bodies.names())
```

A file that exists already is replaced only with `overwrite=True`; otherwise the call raises
`FileExistsError`. The folders of the path are made when needed. The rules of the model file are in
[reference/inputs-model §4](../reference/inputs-model.md#4-the-model-file), and the ways to read a
results file in [reference/results §2](../reference/results.md#2-the-results-file-of-the-engine-hbr).

Programs built on HydroBEM, such as a graphical interface, can keep their own data with the model
in the list `m.tool`, as named texts that the solve does not read:

```python
m.tool.add("my_notes", text="draft 0.5 m, as tested in the basin")
```
