"""The objects of a model in Python: ``ModelObject`` (an object of the
engine's model: its items as attributes, its lists of children), ``ObjectList``, and the meshes,
``MeshObject`` and ``MeshListBase``, with the functions of the engine on them. The class of each
object type, with a property and a description per item and an ``update`` with a keyword per item,
and the class of each list, with an ``add`` with a keyword per item, are generated from the schema
(``_classes.py``, scripts/gen_model_classes.py).

An object of the engine's model is an ``hydrobem._core.Object`` (``ModelObject._o``): Python holds
no copy of the inputs, reading an item reads the engine's, setting one sets it (the engine checks it)."""

from __future__ import annotations

import functools
import json
import numbers
import operator
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from . import _core
from ._core import Object
from .queries import (
    Hydrostatics,
    ItemInfo,
    MeshElements,
    MeshGeometry,
    MeshQuality,
    Panels,
    PatchPoints,
    Tessellation,
    _divisions,
    _elements,
    _flag,
    _geometry,
    _hydrostatics,
    _panels,
    _quality,
    _roles,
    _tessellation,
)

T = TypeVar("T", bound="ModelObject")

# the objects of the root that are always there, and its lists
_SINGLES = ("environment", "options", "first_order", "second_order")


@functools.cache
def schema() -> dict:
    """The schema of the engine: per object type its items (name, kind, dtype, shape,
    unit, default, choices, target, kinds, methods, description) and lists (name, types, named,
    optional_names, single, always, description)."""
    return _core.schema()


def _items(type_name: str) -> dict:
    return schema()[type_name]["items"]


def _lists(type_name: str) -> dict:
    return schema()[type_name]["lists"]


def _shown(v) -> str:
    """A value in a summary: a short array in full, a long one by its shape."""
    if isinstance(v, np.ndarray):
        if v.size <= 6:
            return np.array2string(v, separator=", ", max_line_width=1_000_000)
        return f"array {tuple(v.shape)}"
    if isinstance(v, tuple) and len(v) > 6:
        return f"({', '.join(map(repr, v[:6]))}, ... {len(v)} in all)"
    return repr(v)


def _summary(o: Object, indent: str = "") -> list[str]:
    """The lines of an object's summary: its items given (not their defaults) that its switches
    choose, a line naming the items its switches leave out, and its lists of children, each child
    summarized below it (``print(obj)``)."""
    lines = []
    items = _items(o.type)
    for name in _applying(o):
        if name != "name" and o.describe(name)["is_set"] and _chosen(o, items[name]):
            lines.append(f"{indent}{name} = {_shown(o.get(name))}")
    unused = _unused(o)
    if unused:
        lines.append(f"{indent}unused: {unused}")
    for list_name, info in _lists(o.type).items():
        kids = o.children(list_name)
        if kids and info["single"]:
            body = _summary(kids[0], indent + "  ")
            if body or not info["always"]:
                lines += [f"{indent}{list_name}: {kids[0].type}"] + body
        elif kids:
            lines.append(f"{indent}{list_name}: {len(kids)}")
            for k, c in enumerate(kids):
                lines.append(f"{indent}  [{repr(c.name) if c.name else k}] {c.type}")
                lines += _summary(c, indent + "    ")
    return lines


def _name(name) -> str:
    """The name of a new object: a string."""
    if not isinstance(name, str):
        raise TypeError(f"a name must be a string, got {type(name).__name__}")
    return name


def _default(text: str):
    """The default of an item from the schema's literal ("scattered", 1.5, inf, [0.0, 0.0]); None
    without one."""
    if not text:
        return None
    t = re.sub(r"(?<![\w\"])(-?)inf(?![\w\"])", r'"\1Infinity"', text)
    v = json.loads(t.replace('"Infinity"', "Infinity").replace('"-Infinity"', "-Infinity"))
    return tuple(v) if isinstance(v, list) else v


def _position(key) -> int | None:
    """An integer position (a Python or NumPy integer, not a bool), or None for any other key."""
    if isinstance(key, bool):
        return None
    try:
        return operator.index(key)
    except TypeError:
        return None


def _chosen(o: Object, d: dict) -> bool:
    """Whether the switches of object o choose an item (as the engine's check): one of its
    switch_values ("v", or "v/w" with the second switch) is the value of its switch items; an
    item without a switch always."""
    if not d["switch"]:
        return True
    values = [o.get(sw) for sw in d["switch"]]
    return any(all(w == v for w, v in zip(t.split("/"), values, strict=False)) for t in d["switch_values"])


def _no_item(o: Object, name: str) -> AttributeError:
    """The refusal of a name the object does not have."""
    return AttributeError(f"a {o.type} has no item {name!r}")


def _unused(o: Object) -> str:
    """The items the switches of object o leave out, by switch value: of a switch whose every item
    of its prefix is left out "control_surface_*", a run of three or more "Ixx ... Izz", e.g.
    'control_surface_* (control_surface = "none"), Ixx ... Izz (inertia_input = "radii_of_gyration")';
    "" when the switches leave none out."""
    items = _items(o.type)
    groups: dict[str, list[str]] = {}
    for n, d in items.items():
        if d["switch"] and not _chosen(o, d):
            # the first switch, and the second where the first alone would choose it
            first = o.get(d["switch"][0])
            by = d["switch"][: 1 if all(t.split("/")[0] != first for t in d["switch_values"]) else 2]
            groups.setdefault(" with ".join(f'{sw} = "{o.get(sw)}"' for sw in by), []).append(n)
    parts = []
    for by, names in groups.items():
        switch = by.split(" = ")[0]
        prefix = f"{switch}_"
        of_prefix = [n for n in items if n.startswith(prefix)]
        if of_prefix and set(of_prefix) <= set(names) and all(n.startswith(prefix) for n in names):
            text = f"{prefix}*"
        else:
            order = list(items)
            runs, run = [], [names[0]]
            for n in names[1:]:
                if order.index(n) == order.index(run[-1]) + 1:
                    run.append(n)
                else:
                    runs.append(run)
                    run = [n]
            runs.append(run)
            text = ", ".join(f"{r[0]} ... {r[-1]}" if len(r) >= 3 else ", ".join(r) for r in runs)
        parts.append(f"{text} ({by})")
    return ", ".join(parts)


def _applying(o: Object) -> list[str]:
    """The items that apply to object o: all of its type, of a mesh those of its kind."""
    items = _items(o.type)
    kind = o.get("kind") if o.type == "Mesh" else None
    return [n for n, d in items.items() if not (kind and d["kinds"] and kind not in d["kinds"])]


class ModelObject:
    """An object of a model (a body, a mesh, the options, ...): its items as attributes (the
    schema's names) and its lists of children as ``ObjectList`` attributes. It is a view of an
    object of its model: not copied or pickled (``hb.Model.copy``, a model file)."""

    __slots__ = ("_o",)

    def __init__(self, obj: Object):
        object.__setattr__(self, "_o", obj)

    def __copy__(self):
        raise TypeError(f"a {self._o.type} is an object of its model and is not copied (hb.Model.copy copies the "
                        "model)")  # fmt: skip

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce__(self):
        raise TypeError(f"a {self._o.type} cannot be pickled (save the model as a model file)")

    @property
    def object_type(self) -> str:
        """The type of the object in the schema ("Body", "Mesh", "Options", ...)."""
        return self._o.type

    @property
    def name(self) -> str:
        """The name of the object in its list ("" for an object without a name)."""
        return self._o.name

    @property
    def path(self) -> str:
        """The path of the object in its model: its lists and names from the model down
        ("bodies/hull", "bodies/hull/tanks/1"; "" for the model)."""
        return self._o.path

    def info(self, item: str) -> ItemInfo:
        """What the schema says of an item (kind, unit, default, choices, description, ...) and its
        state (is_set, has_value, the dtype and shape of its value), read-only."""
        d = _items(self._o.type).get(item)
        if d is None:
            raise AttributeError(f"a {self._o.type} has no item {item!r}")
        state = self._o.describe(item)
        return ItemInfo(kind=d["kind"], dtype=d["dtype"], shape=d["shape"], unit=d["unit"],
                        default=_default(d["default"]), choices=tuple(d["choices"]), target=d["target"],
                        kinds=tuple(d["kinds"]), switch=tuple(d["switch"]), switch_values=tuple(d["switch_values"]),
                        description=d["description"], is_set=state["is_set"], has_value=state["has_value"],
                        is_array=state["is_array"], value_dtype=state["dtype"], value_shape=tuple(state["shape"]))

    def reset(self, item: str):
        """Returns an item to its default (or unsets it)."""
        if item not in _items(self._o.type):
            raise AttributeError(f"a {self._o.type} has no item {item!r}")
        self._o.set(item, None)

    def items(self) -> dict:
        """The object's own items that apply, with their values (None: unset), not its name nor its
        lists: ``add("b2", **b1.items())`` copies them."""
        return {name: self._o.get(name) for name in _applying(self._o) if name != "name"}

    def __getattr__(self, name):
        t = self._o.type
        if name in _items(t):
            return self._o.get(name)
        if name in _lists(t):
            return self._child(name)
        raise AttributeError(f"a {t} has no item or list {name!r}")

    def _child(self, name):
        """The list of children, or the child of a single list (None when there is none)."""
        if _lists(self._o.type)[name]["single"]:
            kids = self._o.children(name)
            return _wrap(kids[0]) if kids else None
        return _list_of(self._o, name)

    def __setattr__(self, name, value):
        self._set(name, value)

    def _set(self, name, value):
        """Sets an item; a list of objects is not assigned (its objects are made by add() and
        insert())."""
        t = self._o.type
        if name in _items(t):
            self._o.set(name, value)
            return
        if name in _lists(t):
            raise _no_list_assignment(self._o, name)
        raise _no_item(self._o, name)

    def update(self, **items):
        """Sets several of the object's own items at once, as attributes, all or nothing: when one
        fails, those set before it get their earlier values back. A list of objects is not set here
        (its add() and insert())."""
        t = self._o.type
        for k in items:
            if k in _lists(t):
                raise _no_list_assignment(self._o, k)
            if k not in _items(t):
                raise _no_item(self._o, k)
        # every item, as they are restored together
        before = {k: self._o.get(k) if self._o.describe(k)["is_set"] else None for k in _items(t)}
        try:
            for k, v in items.items():
                self._o.set(k, v)
        except BaseException:
            changed = [k for k in before if not _same(before[k], self._o.get(k) if self._o.describe(k)["is_set"]
                                                      else None)]  # fmt: skip
            for k in changed:  # unset first, then set
                if before[k] is None:
                    self._o.set(k, None)
            for k in changed:
                if before[k] is not None:
                    self._o.set(k, before[k])
            raise

    def _update(self, given: dict):
        """``update`` of the keywords given to a generated ``update``."""
        ModelObject.update(self, **_given(given))

    def __dir__(self):
        t = self._o.type
        return sorted(set(_items(t)) | set(_lists(t)) | {n for n in dir(type(self)) if not n.startswith("_")})

    def __repr__(self):
        if self._o.model.closed:
            return f"<{type(self).__name__} (closed)>"
        return f"<{self._o.type} {self._o.path or 'model'}>"

    def __str__(self):
        head = f"{self._o.type} {self._o.name!r}" if self._o.name else f"{self._o.type} {self._o.path}"
        return "\n".join([head] + _summary(self._o, "  "))

    def __eq__(self, other):
        return isinstance(other, ModelObject) and self._o == other._o

    def __hash__(self):
        return hash(self._o)


class BodyObject(ModelObject):
    """A body of a model: its items, and the functions of the engine on it: what needs its
    placement, trim, mass or the environment (after the rules of the environment, the geometry and
    the surfaces; those of the options, first and second order, frequencies and field points do not concern
    them)."""

    __slots__ = ()

    def hydrostatics(self) -> Hydrostatics:
        """The hydrostatics of the body after its trim and placement (theory §3.2), body axes about
        R: the volumes, the displaced mass, the centre of buoyancy, the waterplane, the stiffness
        without and with the weight terms, and with them the metacentric heights."""
        return _hydrostatics(self._o.body_hydrostatics())

    def panels(self, role=None, include_reflections: bool = True) -> Panels:
        """The elements of the body as the engine uses them (after its trim), placed, for display:
        the elements of its mesh (the lids placed with the body among them, role "lid"), then its
        surfaces of other meshes (the dipoles, interior and pressure surfaces and free-surface
        plates), the control surface, a generated interior surface; the seven keys as
        ``Model.panels``. ``role`` None (every role but "control_surface"), a role name or a list of
        them (the roles of ``Model.panels`` but "fixed_damper" and "free_surface", which are
        elements of no body: ``m.panels(role="fixed_damper")``); ``include_reflections`` also the
        copies about the symmetry planes."""
        return _panels(self._o.body_panels(_flag(include_reflections), _roles(role)))

    def mesh_quality(self) -> MeshQuality:
        """The quality of the body's mesh as the engine uses it: the geometric findings and the
        panels or spans per wavelength (which need the frequencies); the flags of an element are
        those of "orientation", "free_edge", "non_manifold", "coarse" that apply."""
        return _quality(self._o.body_mesh_quality())

    def tessellation(self, role=None, divisions=8, include_reflections: bool = True) -> Tessellation:
        """A quadrilateral mesh of the elements of ``panels`` (the same ``role`` and
        ``include_reflections``), placed, for display, each quad with the keys of its element:
        ``divisions`` quads per patch in u and v (an integer or a pair, 1..256; "spans": one quad
        per span of the potential discretization), one quad per low-order panel."""
        return _tessellation(self._o.body_tessellation(_roles(role), _divisions(divisions), _flag(include_reflections)))

    def trimmed_mesh(self, name: str, role: str = "wetted") -> MeshObject:
        """A new mesh of the model: the surface of one role of this body as the engine uses it, after
        its trim (``role``: "wetted", "dipole", "dipole_damper" (with those inside the tanks),
        "tank", "pressure_surface", "free_surface_plate", "interior_surface"); its
        ``source_elements`` the stored elements of the body's mesh its elements come from. A
        generated interior surface of the low order is the surface the solve generates (its
        ``source_elements`` 0)."""
        return _wrap(self._o.trimmed_mesh(_name(name), str(role)))


class MeshObject(ModelObject):
    """A mesh of a model: its items, and the functions of the engine on it: its geometry alone, in
    its own coordinates (no density, gravity, placement or trim)."""

    __slots__ = ()

    def geometry(self) -> MeshGeometry:
        """The volume closed by the mesh and the plane z = 0 (an open wetted surface closed by its
        waterplane, a closed hull its whole volume): its three estimates by the divergence theorem
        and their mean, the centre of volume, the surface area, the bounds and the waterplane at
        z = 0 (area, centroid, second moments about the origin); over the whole mesh, its
        reflections included."""
        return _geometry(self._o.mesh_geometry())

    def elements(self) -> MeshElements:
        """The stored elements of the mesh, without their reflections: per element (one-based,
        the numbering of the mesh references) its centroid, unit normal, area and vertices (a
        patch: its corners and the values at its centre). A mask over them gives ranges:
        ``hb.ranges(mask)``."""
        return _elements(self._o.mesh_elements())

    def tessellation(self, divisions=8, include_reflections: bool = True) -> Tessellation:
        """A quadrilateral mesh of this mesh alone for display, mesh coordinates (role "", body,
        tank and owner -1, the stored element, part 0): ``divisions`` quads per patch in u and v
        (an integer or a pair, 1..256; "spans": one quad per span of its discretization), one quad
        per low-order panel; ``include_reflections`` also the copies about the symmetry planes."""
        return _tessellation(self._o.mesh_tessellation(_divisions(divisions), _flag(include_reflections)))

    def quality(self) -> MeshQuality:
        """The geometric quality of the mesh alone: per element the panel or span size and aspect ratio
        and the flags "orientation", "free_edge", "non_manifold" (no wavelength: no "coarse"); the
        three volumes and the edge counts (of a patch mesh, its patch sides) of its surface closed by
        the waterline z = 0 (one row, body -1)."""
        return _quality(self._o.mesh_geometry_quality())

    def write(self, gdf, spl=None, overwrite: bool = False) -> None:
        """Writes the mesh, every number at full precision: a path ending in ``.hbg`` the geometry
        file (any kind but a trim, its discretization inside; reference/inputs-meshes §5), otherwise a panels,
        flat or spline patch mesh as GDF (and SPL); the folder is made, and an existing file is kept
        unless ``overwrite``."""
        self._o.write(Path(gdf), None if spl is None else Path(spl), bool(overwrite))

    def to_hbg(self) -> dict:
        """The mesh as the object of a geometry file (.hbg, reference/inputs-meshes §5): the JSON value as dicts
        and lists."""
        return json.loads(self._o.hbg())

    def patch_points(self, element: int, uv) -> PatchPoints:
        """Points and unit normals (into the fluid) of a patch (its one-based element) at
        parameters uv [n, 2] in [-1, 1]², mesh coordinates."""
        uv = np.ascontiguousarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError("uv must have the shape (n, 2)")
        points, normals = self._o.patch_points(int(element), uv)
        return PatchPoints(points, normals)


def _json_default(x):
    """NumPy arrays and scalars in the object of a geometry file."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"a geometry file holds numbers, strings, lists and dicts, got {type(x).__name__}")


def _same(a, b) -> bool:
    """Two item values are equal (arrays element by element, NaN equal to NaN)."""
    if a is None or b is None:
        return a is b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    return a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)


def _given(arguments: dict) -> dict:
    """The keywords given to a generated ``add`` or ``update`` (its ``locals()``): those not left at
    ``...``, the default that stands for "not given"."""
    return {k: v for k, v in arguments.items() if k not in ("self", "name") and v is not ...}


def _list_of(parent: Object, name: str) -> ObjectList:
    """A list of children as the class of its list (generated), for a list of one type."""
    from ._classes import LISTS

    types = _lists(parent.type)[name]["types"]
    return (LISTS.get(types[0], ObjectList) if len(types) == 1 else ObjectList)(parent, name)


def _wrap(obj: Object) -> ModelObject:
    """The object as its class (generated from the schema)."""
    from ._classes import CLASSES

    own = {"Mesh": MeshObject}
    return CLASSES.get(obj.type, own.get(obj.type, ModelObject))(obj)


def _fill(obj: Object, items: dict):
    """The items of a new object; a list of objects is not given here (its add() and insert())."""
    lists = _lists(obj.type)
    for k in items:
        if k in lists:
            raise _no_list_assignment(obj, k)
        if k not in _items(obj.type):
            raise _no_item(obj, k)
    for k, v in items.items():
        obj.set(k, v)
    return obj


def _no_list_assignment(obj: Object, name: str) -> TypeError:
    """The refusal of a list of objects given as a value: its objects are made by add() (or
    insert()) once their parent is made, and the list is emptied by clear()."""
    at = f"{obj.path}: " if obj.path else ""
    return TypeError(f"{at}{name} is not assigned: make its objects with {name}.add(...) (or insert) once their "
                     f"parent is made, and empty it with {name}.clear()")  # fmt: skip


def _clear(obj: Object, name: str):
    """Removes every object of a list, all or nothing (refused, nothing removed, while another
    object refers to one of them)."""
    obj.clear(name)


class ObjectList(Generic[T]):
    """A list of child objects: indexed by name (a list with names) or position, ``add``,
    ``insert``, ``move``, ``del``, ``clear()``, ``len``, iteration, ``names()``, ``rename(old, new)``. In a named list
    every object has a name; in a list with optional names (dipoles, dipole dampers, tanks,
    pressure surfaces, couplings) an object may have one, unique within the list, and is found by it."""

    __slots__ = ("_parent", "_name", "_named", "_names", "_types")

    def __init__(self, parent: Object, name: str):
        self._parent, self._name = parent, name
        info = _lists(parent.type)[name]
        self._named, self._types = info["named"], info["types"]
        self._names = info["named"] or info["optional_names"]  # found by a name

    def __copy__(self):
        raise TypeError(f"the list {self._name} is a view of its model and is not copied (hb.Model.copy copies the "
                        "model)")  # fmt: skip

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce__(self):
        raise TypeError(f"the list {self._name} cannot be pickled (save the model as a model file)")

    def _children(self):
        return self._parent.children(self._name)

    def __len__(self):
        return len(self._children())

    def __iter__(self) -> Iterator[T]:
        return (_wrap(c) for c in self._children())

    def names(self) -> list[str]:
        """The names of the objects of the list, in list order ("" for an object without a name)."""
        return [c.name for c in self._children()]

    def _find(self, key) -> Object:
        """The object of a name (a list with names) or a position; a TypeError for any other key."""
        children = self._children()
        if isinstance(key, str):
            if not self._names:
                raise TypeError(f"{self._name} is indexed by position")
            for c in children:
                if key and c.name == key:
                    return c
            raise KeyError(key)
        p = _position(key)
        if p is None:
            raise TypeError(f"{self._name} is indexed by {'name or ' if self._names else ''}position")
        return children[p]

    def __getitem__(self, key) -> T:
        if isinstance(key, slice):  # a slice: the objects as a Python list
            return [_wrap(c) for c in self._children()[key]]
        return _wrap(self._find(key))

    def __contains__(self, key):
        return self._names and isinstance(key, str) and bool(key) and key in self.names()

    def __delitem__(self, key):
        if isinstance(key, slice):  # the objects of a slice
            for c in self._children()[key]:
                c.remove()
            return
        self._find(key).remove()

    def add(self, name: str | None = None, *, type: str | None = None, **items) -> T:
        """A new object at the end, with its own items (and its surfaces from dicts); the objects of
        its lists are added to them once it is made."""
        return self._insert(-1, name, type, items)

    def _add(self, name, given: dict) -> T:
        """``add`` of the keywords given to a generated ``add``."""
        given = _given(given)
        return self._insert(-1, name, given.pop("type", None), given)

    def insert(self, index: int, name: str | None = None, *, type: str | None = None, **items) -> T:
        """A new object before position ``index``, with its items; ``index`` as in ``list.insert``
        (negative from the end, -1 before the last object; beyond the ends: at the start or the
        end)."""
        p = _position(index)
        if p is None:
            raise TypeError(f"the index of {self._name}.insert must be an integer")
        n = len(self)
        p = min(max(p + n if p < 0 else p, 0), n)
        return self._insert(p, name, type, items)

    def _insert(self, index: int, name, type, items: dict) -> T:
        """A new object at the engine's position ``index`` (-1: at the end), with its items."""
        if self._named and name is None:
            raise TypeError(f"the objects of {self._name} have names")
        o = self._parent.add(self._name, name, type=type, index=index)
        try:
            _fill(o, items)
        except BaseException:
            o.remove()
            raise
        return _wrap(o)

    def clear(self):
        """Removes every object of the list, all or nothing (refused, and nothing removed, while
        another object refers to one of them)."""
        _clear(self._parent, self._name)

    def rename(self, old, new: str):
        """Renames an object (by name or position); in a list with optional names "" removes its name."""
        self._find(old).rename(new)

    def move(self, key, index: int):
        """Moves an object (by name or position) to position ``index`` of the list, the others
        keeping their order (negative from the end, as a Python index). The order is the order of
        the objects in the results and the files (bodies, lids, field points, ...)."""
        o = self._find(key)
        p = _position(index)
        if p is None:
            raise TypeError(f"the index of {self._name}.move must be an integer")
        n = len(self)
        if not -n <= p < n:
            raise IndexError(f"{self._name}.move: index {index} out of range ({n} objects)")
        o.move(p + n if p < 0 else p)

    def __repr__(self):
        if self._parent.model.closed:
            return f"<{self._name} (closed)>"
        return f"<{self._name}: {self.names() if self._names else len(self)}>"


class MeshListBase(ObjectList["MeshObject"]):
    """The meshes of a model (the generated ``MeshList`` adds the typed ``add``): ``add(name,
    kind=..., vertices=..., ...)`` from the items of a mesh, and the meshes the engine makes:
    ``load`` (a GDF or geometry file), ``from_hbg`` (the geometry file's content), ``load_csf`` (a
    control surface), ``load_fdf`` (the panels of a free surface), ``analytic``. The generated control
    surfaces are chosen by a body's item ``control_surface``."""

    def load(self, name: str, path, *, higher_order: bool = False, spl=None, discretization=None, panel_size=None,
             spline_order: int | None = None, outer_quadrature: int | None = None,
             inner_quadrature: int | None = None, zero_trailer: bool = False) -> MeshObject:  # fmt: skip
        """A mesh read from a GDF file (and its SPL file) or a geometry file (``.hbg``, its kind the
        file's; reference/inputs-meshes §5) by the engine's readers (reference/inputs-meshes). With ``panel_size``,
        ``spline_order`` and the Gauss orders ``outer_quadrature``, ``inner_quadrature`` set the
        discretization (None: the engine's rule); a geometry file takes them only when it has no
        discretization."""
        orders = {}
        for key, value in (("spline_order", spline_order), ("outer_quadrature", outer_quadrature),
                           ("inner_quadrature", inner_quadrature)):  # fmt: skip
            if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Integral)):
                raise TypeError(f"{key} must be an integer or None")
            if value is not None and value < 1:
                raise ValueError(f"{key} must be >= 1 (None: the engine's rule)")
            orders[key] = None if value is None else int(value)
        return _wrap(self._model().load_mesh(
            _name(name), Path(path), higher_order=bool(higher_order), zero_trailer=bool(zero_trailer),
            spl=None if spl is None else Path(spl),
            discretization=None if discretization is None else _discretization(discretization),
            panel_size=None if panel_size is None else float(panel_size), **orders))

    def from_hbg(self, name: str, data) -> MeshObject:
        """A mesh from the object of a geometry file (.hbg, reference/inputs-meshes §5): a dict as ``json.load``
        gives it (NumPy arrays and numbers are taken as lists and numbers), or its JSON text."""
        text = data if isinstance(data, str) else json.dumps(data, default=_json_default, allow_nan=False)
        return _wrap(self._model().hbg_mesh(_name(name), text))

    def load_fdf(self, name: str, path, *, symmetry: str = "none") -> MeshObject:
        """The panels of a free-surface file (FDF) as a panels mesh in z = 0, for
        ``second_order.free_surface_mesh`` (free_surface_method "mesh"): ``symmetry`` that of the model the
        file belongs to ("none", "yz", "xz", "xz_and_yz"; the file holds none). Only the panels are
        read (reference/inputs-meshes)."""
        return _wrap(self._model().load_fdf(_name(name), Path(path), str(symmetry)))

    def load_csf(self, name: str, path, *, body=None, zero_trailer: bool = False) -> MeshObject:
        """A control surface read from a CSF file; the automatic surface of a file takes the
        mesh and position of ``body`` (a body of the model, or its name)."""
        return _wrap(self._model().load_csf(_name(name), Path(path), bool(zero_trailer), self._body(body)))

    def analytic(self, name: str, shape: str, discretization, *, parts=None, mapping: str = "uniform", discs=(),
                 reference_length=1.0, symmetry="xz_and_yz", label: str = "", **dimensions) -> MeshObject:  # fmt: skip
        """A named analytic shape (reference/inputs-meshes §2), its quadrant x, y >= 0 as analytic patches, with its
        dimensions as keywords: ``"circular_cylinder"`` (radius, draft), ``"elliptic_cylinder"``
        (a, b, draft), ``"sphere"`` (radius), ``"ellipsoid"`` (a, b, c), ``"barge"`` (half_length,
        half_beam, draft), ``"moonpool_cylinder"`` (radius, draft, moonpool_radius) and
        ``"cylinder_with_dipole_discs"`` (radius, draft, optionally moonpool_radius, and ``discs``:
        per annular dipole disc its outer radius, inner radius and depth). ``parts``: the patches in
        their order, of side, bottom, waterplane (the cylinders), hull, waterplane (the sphere and
        the ellipsoid), end, side, bottom, waterplane (the barge), side, bottom, moonpool_wall,
        moonpool_surface (the moonpool cylinder); by default the closed hull without the
        waterplane and the moonpool surface. ``mapping``: ``"uniform"`` or ``"nonuniform"`` (the
        cylinders, the sphere and the moonpool: profiles with zero slope at the patch ends).
        ``discretization``: (nu, nv, ku, kv, iquo, iqui) for all patches, or one per patch;
        ``symmetry`` as the item of a mesh ("none", "yz", "xz", "xz_and_yz")."""
        dims = [(str(k), float(v)) for k, v in dimensions.items()]
        rows = [[float(x) for x in d] for d in discs]
        if any(len(d) != 3 for d in rows):
            raise ValueError("discs: per disc its outer radius, inner radius and depth")
        parts = [] if parts is None else [str(p) for p in ([parts] if isinstance(parts, str) else parts)]
        return _wrap(self._model().analytic_mesh(_name(name), str(shape), dims, rows, parts, str(mapping),
                                                 _discretization(discretization), float(reference_length),
                                                 str(symmetry), _label(label)))  # fmt: skip

    def _model(self):
        """The engine model of this list (``hydrobem._core.Model``)."""
        return self._parent.model

    def _body(self, body):
        """The object of a body of this model: a Body, its name or its index."""
        if body is None:
            return None
        model = self._model()
        if isinstance(body, str):
            return model.object(f"bodies/{body}")
        if _position(body) is not None:
            bodies = model.object().children("bodies")
            if not -len(bodies) <= _position(body) < len(bodies):
                raise IndexError(f"body: there is no body {_position(body)} (the model has {len(bodies)} bodies)")
            return bodies[_position(body)]
        if not isinstance(body, ModelObject) or body._o.model is not model or body.object_type != "Body":
            raise ValueError("body: a Body of this model, its name or its index")
        return body._o


def _label(label) -> str:
    """The label of a mesh the engine makes ("" for none)."""
    if label is not None and not isinstance(label, str):
        raise TypeError(f"a label must be a string, got {type(label).__name__}")
    return "" if label is None else label


def _discretization(d) -> list:
    """Discretization rows (nu, nv, ku, kv, iquo, iqui) [n, 6]: one row, or one per patch."""
    return np.asarray(d, dtype=np.int32).reshape(-1, 6).tolist()


__all__ = ["BodyObject", "MeshListBase", "MeshObject", "ModelObject", "ObjectList"]
