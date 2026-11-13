"""The answers of the engine's queries as objects: of a mesh its geometry, elements,
tessellation and quality (methods of ``m.meshes[name]``), of a body its hydrostatics, panels,
tessellation and mesh quality (methods of ``m.bodies[name]``), of the model the sizes, the memory
estimate, the point classes, the panels, the tessellation and the mesh quality of everything
(methods of ``hb.Model``), and the queries that need no model (``build_info``, ``quantities``,
``wavenumber``). Each is read-only; its arrays are the engine's, not writable.

Every index is zero-based (-1 none) but the mesh element, one-based as in the ``ranges`` of the
inputs. Lengths are in m, areas in m^2, normals unit vectors; the positions of the model and body
queries are global (those of ``Hydrostatics`` body coordinates), of a mesh alone mesh
coordinates."""

from __future__ import annotations

import datetime as _dt
import functools
import math
import numbers
from dataclasses import dataclass

import numpy as np

from . import _core

STANDARD_GRAVITY = 9.80665  # m/s^2, the engine's default gravity (a test compares it with the schema)


def _frozen(a, dtype=None) -> np.ndarray:
    out = np.array(a, dtype=dtype)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class Hydrostatics:
    """The hydrostatics of a body after its trim and placement (theory §3.2), in body axes: the
    positions in body coordinates (as the input center_of_gravity: the centres and the waterplane
    centroid), the stiffness about the rotation centre R."""

    volumes: np.ndarray  # [3] the estimates of the divergence theorem: the integrals of x n_x, y n_y, z n_z, m^3
    volume: float  # their mean, m^3 (as MeshGeometry.volume)
    displaced_mass: float  # rho V, kg, with V = volumes[0] (the volume of a mass "displaced")
    center_of_buoyancy: np.ndarray  # [3] m, body coordinates
    waterplane_area: float  # m^2
    waterplane_centroid: np.ndarray  # [2] x, y, m, body coordinates
    stiffness_rho_g: np.ndarray  # [6, 6] the hydrostatic stiffness without the weight terms (the geometry alone)
    has_weight: bool  # a centre of gravity and a mass are given: the stiffness has the weight terms
    # the mass in use, kg (rho times volumes[0] less the fluid of its tanks when it is "displaced");
    # None without one
    mass: float | None
    center_of_gravity: np.ndarray | None  # [3] m, body coordinates; None without a centre of gravity
    stiffness: np.ndarray  # [6, 6] the stiffness with the weight terms about R, N/m, N, N m
    # [2] GM roll, pitch: C44, C55 / (rho g volumes[0]), m; None without the weight
    metacentric_height: np.ndarray | None


def _hydrostatics(h: dict) -> Hydrostatics:
    """Hydrostatics of the engine's answer (a dict of its fields)."""
    cg = h["center_of_gravity"]
    return Hydrostatics(
        volumes=_frozen(h["volumes"]),
        volume=h["volume"],
        displaced_mass=h["displaced_mass"],
        center_of_buoyancy=_frozen(h["center_of_buoyancy"]),
        waterplane_area=h["waterplane_area"],
        waterplane_centroid=_frozen(h["waterplane_centroid"]),
        stiffness_rho_g=h["stiffness_rho_g"],
        has_weight=h["has_weight"],
        mass=h["mass"] if math.isfinite(h["mass"]) else None,
        center_of_gravity=None if math.isnan(cg[0]) else _frozen(cg),
        stiffness=h["stiffness"],
        metacentric_height=_frozen(h["metacentric_height"]) if h["has_weight"] else None,
    )


@dataclass(frozen=True, eq=False)
class MeshGeometry:
    """The geometry of a mesh alone, in its own coordinates (no density, gravity or placement), over
    the whole mesh (its reflections included): the volume closed by the mesh and the plane z = 0 (an
    open wetted surface closed by its waterplane, a closed hull its whole volume; theory §3.2)."""

    volumes: np.ndarray  # [3] the estimates of the divergence theorem: the integrals of x n_x, y n_y, z n_z, m^3
    volume: float  # their mean, m^3
    center_of_volume: np.ndarray  # [3] the first moments of the volume over the z estimate, m
    area: float  # the surface area, m^2
    bounds: np.ndarray  # [2, 3] the smallest and largest coordinates (a patch: a 17 x 17 grid of its points), m
    waterplane_area: float  # the waterplane at z = 0 (the integral of -n_z), m^2
    waterplane_centroid: np.ndarray  # [2] x, y, m
    waterplane_moments: np.ndarray  # [3] the second moments about the origin: integrals of x^2, y^2, x y, m^4


def _geometry(g: dict) -> MeshGeometry:
    """MeshGeometry of the engine's answer (a dict of its fields)."""
    return MeshGeometry(
        volumes=_frozen(g["volumes"]),
        volume=g["volume"],
        center_of_volume=_frozen(g["center_of_volume"]),
        area=g["area"],
        bounds=g["bounds"],
        waterplane_area=g["waterplane_area"],
        waterplane_centroid=_frozen(g["waterplane_centroid"]),
        waterplane_moments=_frozen(g["waterplane_moments"]),
    )


@dataclass(frozen=True, eq=False)
class MeshElements:
    """The stored elements of a mesh alone, in its own coordinates, without their reflections;
    N elements."""

    element: np.ndarray  # [N] one-based: the numbering of the mesh references (hb.ranges)
    centroids: np.ndarray  # [N, 3] m (a patch: its point at the centre of its parameters)
    normals: np.ndarray  # [N, 3] unit, into the fluid
    areas: np.ndarray  # [N] m^2
    vertices: np.ndarray  # [N, 4, 3] m (a triangle repeats a vertex; a patch its corners)
    distinct_vertices: np.ndarray  # [N] the distinct vertices among the four corners: 3 or 4


@dataclass(frozen=True, eq=False)
class PatchPoints:
    """Points and unit normals of a patch at parameters uv, mesh coordinates."""

    points: np.ndarray  # [n, 3] m
    normals: np.ndarray  # [n, 3] unit, into the fluid


@dataclass(frozen=True, eq=False)
class FreeSurfacePoints:
    """The points and weights of one zone of the potential QTF's free surface as the solve
    integrates it, global."""

    points: np.ndarray  # [n, 3] m
    weights: np.ndarray  # [n] m^2 (m along the outer circle)


@dataclass(frozen=True, eq=False)
class MotionEquation:
    """The rigid-body matrices of the motion equation (theory §3.1), D x D with D = 6N, body axes
    about each R: inertia (kg, kg m, kg m^2), stiffness (hydrostatic, weight and additional) and
    additional damping; None while ``ready`` is False (a body lacks a centre of gravity, a mass or
    inertia)."""

    inertia: np.ndarray | None
    stiffness: np.ndarray | None
    damping: np.ndarray | None
    ready: bool


@dataclass(frozen=True)
class BodyCounts:
    """The elements of one body after the model rules: the stored elements of its mesh, and the
    elements of the mesh as the engine uses it (after its trim, without the elements no role
    takes) by their roles."""

    stored: int  # the stored elements of its mesh (the numbering of the mesh references)
    # the wetted elements of the mesh as the engine uses it (after a trim), pressure surfaces on it included
    wetted: int
    wetted_with_reflections: int  # the wetted elements with their reflections
    dipole: int  # the thin-sheet elements on its mesh: the dipoles and dipole dampers
    interior_surface: int  # the interior free-surface elements on its mesh; -1 when the surface is generated
    pressure_modes: int  # the pressure modes of the body
    symmetry_copies: int  # the symmetry copies: 1, 2 or 4
    generalized_modes: int  # the generalized modes of the body


@dataclass(frozen=True)
class Counts:
    """The sizes of a model after the model rules, without a solve."""

    bodies: int
    meshes: int  # the meshes the bodies and lids use
    lids: int
    ordinary_frequencies: int  # F: the frequencies other than 0 and inf
    headings: int  # H
    coordinates: int  # D: six per body, the generalized and the pressure coordinates
    radiation_columns: int  # K
    field_points: int  # J: the field points in the sea
    tank_points: int  # the field points in the tanks
    lid_elements: int  # the elements of all lids, with their reflections (patches in a higher-order model)
    pressure_elements: int  # the elements of all prescribed-pressure surfaces, with their reflections
    has_zero_frequency: bool  # omega = 0 is requested
    has_infinite_frequency: bool  # omega = inf is requested
    radiation_mode_ids: tuple[int, ...]  # the coordinate (one-based) of each radiation column
    per_body: tuple[BodyCounts, ...]
    per_lid_elements: tuple[int, ...]  # the element count of each lid, with its reflections


@dataclass(frozen=True)
class MemoryEstimate:
    """The memory a solve will use, in bytes, with the thread counts of the options: the resident
    peak (the pages the solve writes, the working set) and the commit charge (the memory reserved,
    with the whole packing buffers that each thread running a matrix product keeps)."""

    statics: int  # shared by all frequencies: the influence data and the geometry
    per_frequency: int  # one frequency thread with its inner threads (resident)
    outputs: int  # the outputs of all frequencies and the results built from them
    peak: int  # resident: statics + frequency_threads x per_frequency + outputs
    commit: int  # the commit charge: peak and the rest of the packing buffers of the threads


@dataclass(frozen=True, eq=False)
class PointClasses:
    """The verdict of the field-point rules (reference/inputs-model-objects §3) on points."""

    status: tuple[str, ...]  # per point its status, the names of Model.classify_points ("fluid", "tank", ...)
    # [n] the zero-based body of the status (on, near or inside a body, inside its waterplane) or
    # of the tank of a point given to a tank; else -1
    body: np.ndarray
    lid: np.ndarray  # [n] the zero-based lid of a point on a lid; else -1
    # [n] of a point given to a tank the zero-based tank of its body (as results.bodies[b].tanks[t]); else -1
    tank: np.ndarray


@dataclass(frozen=True, eq=False)
class Panels:
    """The elements of the bodies and of no body (lids, fixed dampers, the free surface) as the
    engine uses them (after a trim), for display; N elements or reflected copies. The seven keys
    (body, role, tank, owner, element, part, reflection) identify a row of one answer."""

    vertices: np.ndarray  # [N, 4, 3] m, global (a triangle repeats a vertex; a patch its corners)
    distinct_vertices: np.ndarray  # [N] the distinct vertices among the four corners: 3 or 4
    # [N] the body (among the enabled bodies); -1 an element of no body (a lid not placed with a
    # body, a fixed damper, the free surface)
    body: np.ndarray
    # [N] per element its role ("wetted", "interior_surface", "dipole", "dipole_damper",
    # "pressure_surface", "lid", "tank", "fixed_damper", "control_surface", "free_surface",
    # "free_surface_plate")
    role: np.ndarray
    tank: np.ndarray  # [N] the tank of its body whose fluid it bounds or lies in, else -1
    # [N] the input object that makes it, by its place in its list (a dipole, dipole damper,
    # pressure surface, plate in the body's or tank's list; a lid; a fixed damper; tank walls their
    # tank); -1 for wetted, interior_surface, control_surface, free_surface
    owner: np.ndarray
    # [N] one-based, the element in the numbering of the ranges of the mesh its input names (cut by
    # a trim: the element it was cut from; generated: its own numbering from 1)
    element: np.ndarray
    part: np.ndarray  # [N] the part of an element a trim cut (in the order of the trim), 0 not cut
    reflection: np.ndarray  # [N, 2] sx, sy (1 or -1): the copy about the symmetry planes
    centroids: np.ndarray  # [N, 3] m, global
    normals: np.ndarray  # [N, 3] unit, into the fluid (tank walls into the tank fluid)
    areas: np.ndarray  # [N] m^2


@dataclass(frozen=True, eq=False)
class Tessellation:
    """A quadrilateral mesh of surfaces for display; T quads, each with the keys of the panels row
    of its element (the same call's roles and include_reflections)."""

    vertices: np.ndarray  # [T, 4, 3] m, global (of a mesh alone: mesh coordinates)
    normals: np.ndarray  # [T, 4, 3] unit, into the fluid, at the vertices
    body: np.ndarray  # [T] as Panels.body (-1 also for a mesh alone)
    role: np.ndarray  # [T] as Panels.role ("" for a mesh alone)
    tank: np.ndarray  # [T] as Panels.tank
    owner: np.ndarray  # [T] as Panels.owner
    element: np.ndarray  # [T] as Panels.element (of a mesh alone its stored element)
    part: np.ndarray  # [T] as Panels.part
    reflection: np.ndarray  # [T, 2] as Panels.reflection


@dataclass(frozen=True, eq=False)
class MeshEdges:
    """The edge counts of the meshes per body, or of a mesh alone (of a patch mesh, its patch sides)."""

    inconsistent: np.ndarray  # [B] edges whose two panels (patches) disagree in orientation
    free: np.ndarray  # [B] edges of one panel (patch) off the waterline
    non_manifold: np.ndarray  # [B] edges of more than two panels (patches)
    open_waterline: np.ndarray  # [B] free edges on the waterline of an open surface


@dataclass(frozen=True, eq=False)
class MeshQuality:
    """The quality of the body meshes (N elements, B bodies), or of a mesh alone (one row of
    ``volumes``, ``min_per_wavelength`` and ``edges``, body -1, no wavelength)."""

    body: np.ndarray  # [N] the body index; -1 for a mesh alone
    element: np.ndarray  # [N] one-based, the element as in Panels.element
    part: np.ndarray  # [N] its part after a trim, as in Panels.part
    size: np.ndarray  # [N] m, low order the panel diameter d, higher order the diagonal of a span
    aspect: np.ndarray  # [N] d^2 / 2A, 1 for a square
    per_wavelength: np.ndarray  # [N] panels or spans per shortest wavelength; inf without frequencies (and of a mesh)
    flags: tuple[tuple[str, ...], ...]  # per element the flags "orientation", "free_edge", "non_manifold", "coarse"
    volumes: np.ndarray  # [B, 3] the divergence theorem in x, y, z
    min_per_wavelength: np.ndarray  # [B]
    edges: MeshEdges


@dataclass(frozen=True)
class BuildInfo:
    """The build of the engine library."""

    version: str
    commit: str  # git describe of the source
    compiler: str
    linear_algebra: str  # the linear-algebra library and its version (faer)
    build_type: str
    api_version: int
    avx2: bool


@dataclass(frozen=True)
class QuantityInfo:
    """A result quantity of the engine."""

    name: str
    code: int  # the row of the quantity table
    dtype: str  # the NumPy element type
    dims: tuple[str, ...]  # the axis names
    unit: str  # SI; r_i = 1 for a rotation coordinate
    description: str


@dataclass(frozen=True)
class ItemInfo:
    """What the schema says of an item of an object, and its state (``obj.info(item)``)."""

    kind: str  # "real", "int", "bool", "text", "choice", "array", "reference", "origin", "terms"
    dtype: str  # the element type of an array item: "real", "int" or "text" ("real" for a scalar)
    shape: str  # the schema's shape of an array item ("N,4,3"), "" for a scalar
    unit: str
    default: object  # the default value, None without one
    choices: tuple[str, ...]  # the names it takes (a choice, or a number item's names)
    target: str  # the list a reference names ("meshes", "bodies"), ""
    kinds: tuple[str, ...]  # the mesh kinds it applies to, () for all
    switch: tuple[str, ...]  # the switch items that choose it ("control_surface"), () for none
    switch_values: tuple[str, ...]  # their values that take it ("cylinder", "automatic/circle"), () always
    description: str
    is_set: bool  # given (not its default)
    has_value: bool  # given or with a default
    is_array: bool  # its value is an array
    value_dtype: str  # the element type of its value: of an array its elements, "real" for a scalar
    value_shape: tuple[int, ...]  # the shape of its value


@dataclass(frozen=True)
class Meta:
    """The meta data of the results of a solve."""

    hydrobem_version: str
    api_version: int
    commit: str
    compiler: str
    linear_algebra: str
    build_type: str
    solved_at: object  # datetime (UTC); the text when it is not one
    route: str
    frequency_threads: int
    inner_threads: int
    engine_seconds: tuple[float, ...]
    environment_hooks: tuple[str, ...]  # the hooks of the environment set for the solve (internal
    # studies and tests only), NAME or NAME=value; empty without


def _meta(info: dict | None) -> Meta | None:
    """Meta of the engine's meta data (a dict of its fields); None for results not from a solve
    (``Results.of``)."""
    if info is None:
        return None
    info = dict(info)
    try:
        info["solved_at"] = _dt.datetime.fromisoformat(info["time"].replace("Z", "+00:00"))
    except ValueError:  # a time the engine did not write: kept as text
        info["solved_at"] = info["time"]
    del info["time"]
    return Meta(**info)


def build_info() -> BuildInfo:
    """The build of the engine library."""
    return BuildInfo(**_core.build_info())


@functools.cache
def _quantity_table() -> tuple[dict, ...]:
    """The rows of the engine's quantity table, in code order."""
    return tuple(_core.quantities())


@functools.cache
def quantity_names() -> tuple[str, ...]:
    """The names of the result quantities, in the order of their codes (the engine's table)."""
    return tuple(q["name"] for q in _quantity_table())


def quantity_code(name: str) -> int:
    """The code of a quantity name (ValueError for an unknown one)."""
    try:
        return quantity_names().index(name)
    except ValueError:
        raise ValueError(f"no result quantity {name!r}") from None


def quantities() -> dict[str, QuantityInfo]:
    """The result quantities of the engine by name ("added_mass", ...), in code order."""
    return {q["name"]: QuantityInfo(code=k, **q) for k, q in enumerate(_quantity_table())}


def wavenumber(omega, depth=float("inf"), gravity=STANDARD_GRAVITY) -> float:
    """Wave number k of the dispersion relation omega² = g k tanh(k h).

    Parameters
    ----------
    omega : float
        Radian frequency in rad/s (0 gives 0).
    depth : float, default inf
        Water depth h in m.
    gravity : float, default 9.80665
        g in m/s².

    Returns
    -------
    float
        k in 1/m.
    """
    return _core.wavenumber(float(omega), float(depth), float(gravity))


def _role_names(values) -> np.ndarray:
    """The role names of role values (-1: "", a mesh alone)."""
    return _frozen([_core.PANEL_ROLES[r] if r >= 0 else "" for r in values], dtype=np.str_)


def _panels(q: dict) -> Panels:
    """Panels of the engine's answer (a dict of its read-only arrays, the roles by value)."""
    return Panels(**{**q, "role": _role_names(q["role"])})


def _elements(q: dict) -> MeshElements:
    """MeshElements of the engine's answer (a dict of its read-only arrays)."""
    return MeshElements(**q)


def _roles(role) -> list[str]:
    """The roles of a role argument: None (the default roles), a name or a non-empty list of names
    (an empty list, a filter that matched nothing, is refused rather than taken as the default)."""
    if role is None:
        return []
    if isinstance(role, str):
        return [role]
    roles = list(role)
    if not all(isinstance(r, str) for r in roles):
        raise TypeError("role must be None, a role name or a list of role names")
    if not roles:
        raise ValueError("role is an empty list: give None for the default roles, or the roles")
    return roles


def _flag(include_reflections) -> bool:
    """include_reflections: a bool."""
    if not isinstance(include_reflections, (bool, np.bool_)):
        raise TypeError("include_reflections must be True or False")
    return bool(include_reflections)


def _divisions(divisions) -> tuple[int, int]:
    """The quads per patch of a tessellation in u and v: an integer, a pair, or "spans" (0, 0)."""
    if isinstance(divisions, str):
        if divisions != "spans":
            raise ValueError(f'divisions must be an integer, a pair or "spans", got {divisions!r}')
        return 0, 0
    pair = (divisions, divisions) if isinstance(divisions, numbers.Integral) else tuple(divisions)
    if len(pair) != 2 or any(isinstance(n, bool) or not isinstance(n, numbers.Integral) for n in pair):
        raise TypeError('divisions must be an integer, a pair of integers or "spans"')
    if any(n < 1 for n in pair):
        raise ValueError('divisions must be in 1..256 (or "spans")')
    return int(pair[0]), int(pair[1])


def _tessellation(q: dict) -> Tessellation:
    """Tessellation of the engine's answer (a dict of its read-only arrays, the roles by value)."""
    return Tessellation(**{**q, "role": _role_names(q["role"])})


def _quality(q: dict) -> MeshQuality:
    """MeshQuality of the engine's answer (a dict of its read-only arrays, the flags as bits)."""
    edges = q["edges"]
    return MeshQuality(
        **{
            k: q[k]
            for k in ("body", "element", "part", "size", "aspect", "per_wavelength", "volumes", "min_per_wavelength")
        },
        flags=tuple(tuple(k for k, bit in _core.QUALITY_FLAGS.items() if f & bit) for f in q["flags"]),
        edges=MeshEdges(*(_frozen(edges[:, i]) for i in range(4))),
    )


__all__ = [
    "BodyCounts",
    "BuildInfo",
    "Counts",
    "Hydrostatics",
    "ItemInfo",
    "MemoryEstimate",
    "FreeSurfacePoints",
    "MeshEdges",
    "MeshElements",
    "MeshGeometry",
    "MeshQuality",
    "Meta",
    "MotionEquation",
    "Panels",
    "PatchPoints",
    "PointClasses",
    "QuantityInfo",
    "Tessellation",
    "build_info",
    "quantities",
    "wavenumber",
]
