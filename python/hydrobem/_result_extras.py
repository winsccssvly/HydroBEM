"""What the classes of the results have beyond the views of the result tree (reference/results
): the bases of the generated views (``_View``, ``_OwnerView``), the owner lists, and the
hand-written parts of the generated public classes, each a base they name (``_ResultsExtras`` of
``Results``, ``_BodyExtras`` of ``BodyResults``; ``scripts/gen_result_views.py`` maps them). The
public classes are defined in ``_result_views`` alone; this module does not import it."""

from __future__ import annotations

import datetime as _dt
import numbers
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .errors import Diagnostic
from .queries import Hydrostatics, Meta, _meta, _quantity_table, quantity_code, quantity_names

MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
_KINDS = ("rigid", "generalized", "pressure")


@dataclass(frozen=True)
class Coordinate:
    """One coordinate of the D axis (body-major: six rigid modes, the body's generalized modes, then
    its pressure modes)."""

    id: int  # one-based global id
    body: int  # body index
    kind: str  # "rigid", "generalized" or "pressure"
    local: int  # rigid mode 1..6, or the generalized or pressure mode index (one-based) within the body
    name: str
    rotational: bool


def _position(key, n: int, what: str) -> int:
    """The zero-based position of an index (negative from the end) among n; IndexError beyond."""
    if isinstance(key, bool) or not isinstance(key, numbers.Integral):
        raise TypeError(f"{what} is indexed by name or position, not {type(key).__name__}")
    k = int(key)
    if not -n <= k < n:
        raise IndexError(f"{what} {k} does not exist ({n} in these results)")
    return k % n


def _find(names: list[str], key, what: str, disabled=()) -> int:
    """The position of a name or an index among names; KeyError for an unknown name (and for a
    name of ``disabled``, the owners not solved)."""
    if isinstance(key, str):
        if key and key in names:
            return names.index(key)
        if key and key in disabled:
            raise KeyError(f"{what}: {key!r} was disabled when the model was solved: these results have none of it")
        raise KeyError(f"no {what} {key!r} in these results ({', '.join(repr(x) for x in names if x) or 'none named'})")
    return _position(key, len(names), what)


def _take(a, axis: int, rows):
    """The rows of an array along an axis, as the engine's selection gives them: a view for a slice
    (contiguous rows), else a read-only copy of the rows of an index array; None for None."""
    if a is None:
        return None
    if isinstance(rows, slice):
        index = [slice(None)] * a.ndim
        index[axis] = rows
        return a[tuple(index)]
    out = np.take(a, np.asarray(rows, dtype=np.intp), axis=axis)
    out.flags.writeable = False
    return out


class _OwnerList:
    """The views of the owners of one kind (bodies, lids, the pressure surfaces or the tanks of a
    body): by name or position, ``len``, iteration, ``names()``; ``name in`` the list tests a name."""

    __slots__ = ("_names", "_make", "_what", "_disabled")

    def __init__(self, names, make, what, disabled=()):
        self._names, self._make, self._what, self._disabled = list(names), make, what, tuple(disabled)

    def __len__(self):
        return len(self._names)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._make(k) for k in range(len(self._names))[key]]
        return self._make(_find(self._names, key, self._what, self._disabled))

    def __iter__(self):
        return (self._make(k) for k in range(len(self._names)))

    def __contains__(self, key):
        return isinstance(key, str) and bool(key) and key in self._names

    def names(self) -> list[str]:
        """The names in order ("" for an owner without a name)."""
        return list(self._names)

    def __repr__(self):
        return f"<{self._what}: {self._names}>"


class _View:
    """A view of results for an owner: the results, the zero-based body (None: the results) and the
    zero-based owner within its list (a lid, a pressure surface or a tank; None for the others)."""

    __slots__ = ("_res", "_body", "_owner")

    def __init__(self, res, body=None, owner=None):
        self._res, self._body, self._owner = res, body, owner

    def _get(self, path):
        return self._res._select(path, self._body, self._owner)

    def __repr__(self):
        return f"<{type(self).__name__}{'' if self._body is None else f' of body {self._body}'}>"


class _OwnerView(_View):
    """The view of one owner of a list (a lid, a pressure surface, a tank): ``index`` its zero-based
    position in the list, ``name`` its name ("" for none)."""

    __slots__ = ("index", "name")

    def __init__(self, res, body, owner, name):
        super().__init__(res, body, owner)
        self.index, self.name = owner, name

    def __repr__(self):
        return f"<{type(self).__name__} {self.name or self.index!r}>"


class _BodyExtras(_View):
    """The hand-written part of ``BodyResults``: ``index`` and ``name`` of the body, its
    ``coordinates`` and ``hydrostatics``."""

    __slots__ = ("index", "name", "_d", "_k")

    def __init__(self, res, body: int, owner=None, name: str = ""):
        super().__init__(res, body)
        self.index, self.name = body, name
        self._d, self._k = res._body_columns(body)

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        """[D_b] its coordinates (``hb.Coordinate``)."""
        c = self._res.coordinates
        return tuple(c[d] for d in self._d)

    @property
    def hydrostatics(self) -> Hydrostatics | None:
        """Its hydrostatics of the solve (``hb.Hydrostatics``, as ``body.hydrostatics()``), from the
        ``hydrostatics_*`` quantities; None without them."""
        return self._res._hydrostatics(self.index)

    def __repr__(self):
        return f"<body results {self.name!r}: {len(self._d)} coordinates, {len(self._k)} radiation columns>"


class _ResultsExtras:
    """The hand-written part of ``Results``: the engine's results it holds, a quantity by name, what
    the results are of, the coordinates, the owner lists, the coefficients by coordinate and the
    files."""

    __slots__ = ("_r", "_cache", "_model")

    def __init__(self, results: _core.Results):
        """Takes the engine's results."""
        self._r, self._cache, self._model = results, {}, None  # _model: the model as it was solved

    def __reduce__(self):
        raise TypeError("results cannot be pickled (save them as a results file)")

    def quantity(self, name: str):
        """The array of a quantity by its engine name (``hb.quantities()``), whole: for code that takes
        the name from a list, and for the quantities without a view of the results (``coordinates``,
        the owner numbers ``lids_ids``, ``pressure_surfaces_ids``, ``tanks_ids``, the whole arrays of
        the lids, pressure surfaces and tanks: ``lids``, ``pressure_surfaces_elevation_radiation``,
        ``tanks_pressure_radiation``, ...); None when it is not available, ValueError for an unknown
        name."""
        return self._quantity(quantity_code(name))

    def _quantity(self, q: int):
        if q not in self._cache:
            self._cache[q] = self._r.quantity(q)
        return self._cache[q]

    def _get(self, path):
        return self._select(path)

    @classmethod
    def of(cls, model, **quantities):
        """Results of the present inputs of a model, given as arrays by quantity name (results computed
        elsewhere, to write them with ``write_wamit``). The axes (``omegas``, ``headings``,
        ``radiation_mode_ids``, ``coordinates``) and the points of the field outputs
        (``field_points``, ``tanks``, ``tanks_ids``) are the model's and are not given; every array
        must have the shape of its axes on the model (the extent of each other axis, lid points,
        body points, angles, the same in every quantity) and is converted to the quantity's element
        type; a real quantity takes no complex values.

        The quantities read together are given together (a ValueError names the one missing):
        the eleven ``hydrostatics_*`` all or none; ``drift_pressure_integration`` with
        ``drift_pressure_integration_body_axes``, and ``drift_control_surface`` with
        ``drift_control_surface_body_axes`` (each needs the other); any drift
        (``drift_pressure_integration``, ``drift_control_surface``, ``drift_momentum_conservation``)
        with ``drift_heading_pairs``; ``body_points_ids`` with a body in range, an element >= 1 and
        sx, sy in -1..1 in each row, and when a row is of a panel (sx or sy not 0) with
        ``body_points``, ``body_points_normals`` and ``body_points_areas``."""
        given = []
        for name, value in quantities.items():
            dtype = np.dtype(_quantity_table()[quantity_code(name)]["dtype"])
            if np.iscomplexobj(value) and dtype.kind != "c":
                raise TypeError(f"{name} is real, got complex values")
            given.append((name, np.ascontiguousarray(value, dtype=dtype)))
        return cls(model._doc.new_results(given))

    # ---------------------------------------------------------------- what the results are of

    @property
    def model(self):
        """The model solved (``hb.Model``): a copy of the inputs of these results, every object (the
        disabled ones with ``enabled`` false)."""
        from .engine import Model

        return Model(self._r.model())

    def _inputs(self):
        """The model as it was solved: its enabled objects, which the quantities number (the views
        by owner read it)."""
        if self._model is None:
            from .engine import Model

            self._model = Model(self._r.solved_model())
        return self._model

    def _disabled_bodies(self) -> tuple[str, ...]:
        """The names of the bodies of the inputs that were not solved (disabled)."""
        solved = set(self._inputs().bodies.names())
        return tuple(n for n in self.model.bodies.names() if n not in solved)

    @property
    def meta(self) -> Meta | None:
        """The meta data of the solve: the build, the time, the route, the threads and the engine
        seconds (read-only); None for results not from a solve (``Results.of``)."""
        return _meta(self._r.meta())

    @property
    def route(self) -> str:
        """The solver route of the solve ("low_order_infinite_open_sea", ...)."""
        return self._r.route

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        """The warnings of the model and of the solve, each also issued as ``HydroBEMWarning``
        (``diagnostics``: every severity)."""
        return tuple(d for d in self._r.diagnostics() if d.severity == "warning")

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Every diagnostic of the results, of every severity: the warnings of the model and of
        the solve, and the errors a results file holds (a ``Diagnostic`` each)."""
        return tuple(self._r.diagnostics())

    @property
    def available(self) -> tuple[str, ...]:
        """The names of the quantities the results have, in the order of the quantity table
        (``hb.quantities()``)."""
        names = quantity_names()
        return tuple(names[k] for k in self._r.available())

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        """The coordinates of the D axis, from the engine's ``coordinates`` and the names in the model."""
        c = self.quantity("coordinates")
        if c is None:
            return ()
        m = self._inputs()
        out = []
        for i, (body, kind, local, rot) in enumerate(np.asarray(c).tolist()):
            b = m.bodies[body]
            if kind == 0:
                name = MODE_NAMES[local - 1]
            elif kind == 1:
                name = b.generalized_modes[local - 1].name
            else:
                name = [mode.name for ps in b.pressure_surfaces for mode in ps.pressure_modes][local - 1]
            out.append(Coordinate(i + 1, body, _KINDS[kind], local, name, bool(rot)))
        return tuple(out)

    def normalized(self, name: str) -> np.ndarray:
        """A quantity normalized as in the numeric files (reference/file-formats §1, by rho, g, L and
        omega): float64 or complex128, the shape of the quantity."""
        q = quantity_code(name)
        if self._quantity(q) is None:
            raise ValueError(f"{name} is not available in these results")
        return self._r.normalized(q)

    def summary(self) -> str:
        """A short text listing what was computed and the array shapes."""
        def n(name):
            a = self.quantity(name)
            return 0 if a is None else len(a)

        def count(k, one, many):
            return f"{k} {one if k == 1 else many}"

        sizes = (count(len(self._inputs().bodies), "body", "bodies"),
                 count(n("omegas"), "ordinary frequency", "ordinary frequencies"),
                 count(n("headings"), "heading", "headings"),
                 count(len(self.coordinates), "coordinate", "coordinates"),
                 count(n("radiation_mode_ids"), "radiation column", "radiation columns"))  # fmt: skip
        lines = [f"HydroBEM results ({self.route}): " + ", ".join(sizes)]
        for q, name in enumerate(quantity_names()):
            if name in ("omegas", "headings", "radiation_mode_ids", "coordinates"):
                continue
            a = self._quantity(q)
            if a is not None:
                lines.append(f"  {name}: {a.shape}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- by owner and by name

    def _body_columns(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        """The zero-based D indices of a body's coordinates and the K indices of its radiation
        columns (the engine's)."""
        return self._r.body_columns(body)

    def _select(self, path: str, body=None, owner=None):
        """The view of a path of the result tree for an owner (the engine's selection, applied to
        the cached array): the array, a slice of it, a read-only copy of rows or None."""
        s = self._r.selection(path, body, owner)
        if s is None:
            return None
        code, axes = s
        a = self._quantity(code)
        for axis, rows in axes:
            a = _take(a, axis, rows)
        return a

    def _owner_list(self, path: str, body, make) -> _OwnerList:
        """The list of the owners of a path ("bodies" and "lids" of the results, "pressure_surfaces"
        or "tanks" of a body), by the names in the model solved; ``make`` the class of the view of
        one, called with the results, the body, the owner within the list and its name."""
        if path == "bodies":
            names = self._inputs().bodies.names()
            return _OwnerList(names, lambda k: make(self, k, None, names[k]), "bodies", self._disabled_bodies())
        if path == "lids":
            names, what = self._inputs().lids.names(), "lids"
        elif path in ("pressure_surfaces", "tanks"):
            b = self._inputs().bodies[body]
            names = getattr(b, path).names()
            what = f"{path.replace('_', ' ')} of {b.name}"
        else:
            raise ValueError(f"no owner list {path!r}")
        return _OwnerList(names, lambda k: make(self, body, k, names[k]), what)

    def _hydrostatics(self, body: int) -> Hydrostatics | None:
        """The hydrostatics of a body from the ``hydrostatics_*`` quantities (as the engine's query)."""
        names = ("volumes", "volume", "displaced_mass", "center_of_buoyancy", "waterplane_area",
                 "waterplane_centroid", "stiffness_rho_g", "mass", "center_of_gravity", "stiffness",
                 "metacentric_height")  # fmt: skip
        q = {k: self.quantity(f"hydrostatics_{k}") for k in names}
        if any(v is None for v in q.values()):
            return None
        mass, cg, gm = float(q["mass"][body]), q["center_of_gravity"][body], q["metacentric_height"][body]
        weight = not np.isnan(mass) and not np.isnan(cg[0])
        return Hydrostatics(
            volumes=q["volumes"][body],
            volume=float(q["volume"][body]),
            displaced_mass=float(q["displaced_mass"][body]),
            center_of_buoyancy=q["center_of_buoyancy"][body],
            waterplane_area=float(q["waterplane_area"][body]),
            waterplane_centroid=q["waterplane_centroid"][body],
            stiffness_rho_g=q["stiffness_rho_g"][body],
            has_weight=weight,
            mass=None if np.isnan(mass) else mass,
            center_of_gravity=None if np.isnan(cg[0]) else cg,
            stiffness=q["stiffness"][body],
            metacentric_height=gm if weight else None,
        )

    def coordinate(self, body, mode) -> tuple[int, int | None]:
        """The zero-based index of a coordinate on the D axis and on the K axis (None when it is no
        radiation column): ``body`` by name or index, ``mode`` by name ("heave", a generalized or a
        pressure mode name) or by its one-based position among the body's coordinates."""
        names = self._inputs().bodies.names()
        b = _find(names, body, "body", self._disabled_bodies() if isinstance(body, str) and body not in names else ())
        d, _ = self._body_columns(b)
        mine = [self.coordinates[i] for i in d]
        if isinstance(mode, str):
            found = [i for i, c in zip(d, mine, strict=True) if c.name == mode]
            if not found:
                raise KeyError(f"body {names[b]!r} has no mode {mode!r} (its modes: "
                               f"{', '.join(c.name for c in mine)})")  # fmt: skip
            di = int(found[0])
        elif isinstance(mode, bool) or not isinstance(mode, numbers.Integral):
            raise TypeError(f"a mode is a name or a one-based position, not {type(mode).__name__}")
        elif not 1 <= mode <= len(d):
            raise IndexError(f"body {names[b]!r} has the modes 1..{len(d)}, not {mode}")
        else:
            di = int(d[mode - 1])
        ids = self.quantity("radiation_mode_ids")
        k = None if ids is None else np.flatnonzero(np.asarray(ids) == di + 1)
        return di, (int(k[0]) if k is not None and k.size else None)

    def coefficient(self, name: str, row, column=None) -> np.ndarray:
        """One series of a quantity by coordinates, each a ``(body, mode)`` pair as ``coordinate``
        takes: of an [F, D, K] quantity (``added_mass``, ...) the [F] series of a row and a radiation
        column; of an [F, H, D] or [F, H, K] quantity (``excitation_diffraction``, ...) the [F, H]
        series of a row (``column`` None); of a second-order quantity on the coordinates
        (``drift_pressure_integration`` [F, P, D], ...) the series of a row; likewise the element of
        a [D, K] or [D, D] quantity (a 0-d array). Read-only, as every array of the results."""
        code = quantity_code(name)
        axes = _quantity_table()[code]["dims"]
        a = self._quantity(code)
        if a is None:
            raise ValueError(f"{name} is not available in these results")
        coords = [i for i, x in enumerate(axes) if x in ("coordinate", "radiation_column")]
        if not coords or coords[0] != len(axes) - len(coords):
            raise ValueError(f"{name} is not indexed by coordinates (its axes: {', '.join(axes)})")
        given = [row] if column is None else [row, column]
        if len(given) != len(coords):
            raise TypeError(f"{name} takes {'a row and a column' if len(coords) == 2 else 'a row alone'}")
        index = [slice(None)] * (len(axes) - len(coords))
        for axis, coord in zip(coords, given, strict=True):
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise TypeError("a coordinate is a (body, mode) pair")
            d, k = self.coordinate(*coord)
            if axes[axis] == "radiation_column":
                if k is None:
                    raise ValueError(f"{coord} is not a radiation column of these results")
                index.append(k)
            else:
                index.append(d)
        out = a[tuple(index)]
        if not isinstance(out, np.ndarray):  # the element of a [D, K] or [D, D] quantity
            out = np.array(out)
            out.flags.writeable = False
        return out

    # ---------------------------------------------------------------- files

    def save(self, path, overwrite=False) -> None:
        """Write the results file of the engine (.hbr): the model, the meta data and every
        quantity; ``hb.load_results(path)`` reads it back. The folder is made."""
        self._r.save(Path(path), bool(overwrite))

    def write_wamit(self, stem, frequency="period", timestamp=None, overwrite=False) -> None:
        """Write every computed quantity as WAMIT-format numeric files (.1, .2, .3, .4, .5*, .6*, .7,
        .8, .9, .hst, .fpt, .pnl) and the .out summary (reference/file-formats §2-§4; the engine writes them).

        Parameters
        ----------
        stem : str or Path
            Output path without extension, e.g. "out/cylinder"; the folder is made.
        frequency : {"period", "omega"}, default "period"
            First column of the numeric files: period in s or omega in rad/s.
        timestamp : datetime, str or None, default None
            The header time stamp; None: now.
        overwrite : bool, default False
            Replace existing files.

        """
        if frequency not in ("period", "omega"):
            raise ValueError("frequency must be 'period' or 'omega'")
        if isinstance(timestamp, _dt.datetime):
            timestamp = timestamp.replace(microsecond=0).isoformat(sep=" ")
        elif timestamp is not None and not isinstance(timestamp, str):
            raise TypeError("timestamp must be None, a datetime or a string")
        self._r.write_wamit(Path(stem), overwrite=bool(overwrite), omega=frequency == "omega", stamp=timestamp)
