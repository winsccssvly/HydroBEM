"""The engine model in Python: ``hb.Model`` wraps an engine model
(``hydrobem._core.Model``). Python holds no copy of the inputs: reading an attribute reads
the engine's item, setting one sets it. The attributes come from the engine's schema. The model
holds the results of its last solve.

    m = hb.Model.load("barge.hbm")
    m.bodies["hull"].mass = 1.2e6
    m.first_order.motion = True
    r = m.solve()
    r.write_wamit("out/barge")
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from pathlib import Path

import numpy as np

from . import _core
from ._classes import MeshList, ModelLists
from .errors import HydroBEMWarning
from .objects import (
    _SINGLES,
    BodyObject,
    MeshObject,
    ModelObject,
    ObjectList,
    _list_of,
    _lists,
    _no_list_assignment,
    _summary,
    _wrap,
    schema,
)
from .queries import (
    BodyCounts,
    Counts,
    FreeSurfacePoints,
    MemoryEstimate,
    MeshQuality,
    MotionEquation,
    Panels,
    PointClasses,
    Tessellation,
    _divisions,
    _flag,
    _frozen,
    _panels,
    _quality,
    _roles,
    _tessellation,
)
from .results import Results


class Model(ModelLists):
    """A model held by the engine.

    Its objects are attributes: ``environment`` (with the waves), ``options``, ``first_order``
    (what the first-order solve computes, with the field points in the sea), ``second_order`` (the
    drifts and the QTFs), and the lists ``bodies``, ``lids``, ``fixed_dampers``, ``meshes`` (an
    ``ObjectList``, indexed by name) and ``tool``. Every item is the engine's: it is checked
    when it is set (the value alone), and the rules between items and objects when the model is
    checked or solved.

    Examples
    --------
    >>> m = hb.Model.load("case.hbm")
    >>> m.environment.depth = 50.0
    >>> results = m.solve()
    """

    __slots__ = ("_doc", "_results")

    def __init__(self, _doc: _core.Model | None = None):
        self._doc = _doc if _doc is not None else _core.Model()
        self._results = None  # the Results of the last solve, made when first asked for

    def __setattr__(self, name, value):
        # a list of the model is set from a list of dicts (m.bodies = [...]); nothing else is an attribute
        if name in ("_doc", "_results"):
            object.__setattr__(self, name, value)
        elif name in _lists("Model") and name not in _SINGLES:
            raise _no_list_assignment(self._doc.object(), name)
        elif name in _SINGLES:
            raise AttributeError(f"m.{name} is always there: set its items (m.{name}.update(...))")
        else:
            raise AttributeError(f"a Model has no {name!r}")

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __reduce__(self):
        raise TypeError("a Model cannot be pickled (save it as a model file: m.save)")

    # ---- life cycle
    @classmethod
    def load(cls, path) -> Model:
        """A model file (.hbm), or the inputs of a results file (.hbr), read by the engine."""
        return cls(_core.Model.load(Path(path)))

    def save(self, path, overwrite: bool = False) -> None:
        """Writes the model file (.hbm): every object and item, the meshes inside, in one file; the
        folder is made. ``m.save_results`` writes the model with its results (.hbr)."""
        self._doc.save(Path(path), bool(overwrite))

    def copy(self) -> Model:
        """A new model of a copy of the inputs, without the results."""
        return Model(self._doc.copy())

    # ---- objects
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _lists("Model"):
            return self._child(name)
        raise AttributeError(f"a Model has no {name!r}")

    def _child(self, name):
        """An object of the model (environment, options, first_order, second_order) or a list."""
        root = self._doc.object()
        if name in _SINGLES:
            return _wrap(root.children(name)[0])
        return _list_of(root, name)

    def __dir__(self):
        return sorted(set(_lists("Model")) | set(_SINGLES) | {n for n in dir(type(self)) if not n.startswith("_")})

    def __repr__(self):
        if self._doc.closed:
            return "<hb.Model (closed)>"
        root = self._doc.object()
        return f"<hb.Model: {len(root.children('bodies'))} bodies, {len(root.children('meshes'))} meshes>"

    def __str__(self):
        """Every object of the model with the items given (not their defaults), meshes by kind and
        size."""
        root = self._doc.object()
        lines = ["hb.Model"]
        for name in _lists("Model"):
            kids = root.children(name)
            if name in _SINGLES:
                body = _summary(kids[0], "    ") if kids else []
                if body:
                    lines += [f"  {name}"] + body
            elif kids:
                lines.append(f"  {name}: {len(kids)}")
                for c in kids:
                    lines.append(f"    {c.name!r}")
                    lines += _summary(c, "      ")
        return "\n".join(lines)

    # ---- checks, queries, the solve
    def check(self) -> list:
        """The rules between items and objects and the model rules, then the refusals of the
        solver routes (and with the potential QTF the zones of its free surface, as the solve
        makes them); the first broken rule is raised (``ValueError``), a valid model without a
        route raises ``UnsupportedProblemError`` as ``route`` and the solve do; otherwise the
        warnings are returned (each a ``Diagnostic``)."""
        return self._doc.check()

    @property
    def warnings(self) -> tuple:
        """The warnings of the last rule check (``check``, a solve or a query), located in the
        model, each a ``Diagnostic``; after ``check`` also those of the disabled objects."""
        return tuple(self._doc.warnings())

    def route(self) -> str:
        """The solver route of the model, e.g. "low_order_infinite_open_sea", after the model rules
        (``check`` applies the same): an UnsupportedProblemError names what the engine cannot solve."""
        return self._doc.route()

    def memory_estimate(self) -> MemoryEstimate:
        """The memory a solve will use, in bytes, without solving, with the thread counts
        of the options. Not included: the problems of internal tanks and the copies of the
        results in Python (about the outputs once more)."""
        return MemoryEstimate(**self._doc.memory_estimate())

    def counts(self) -> Counts:
        """The sizes of the model after the model rules, without a solve."""
        c = self._doc.counts()
        c["per_body"] = tuple(BodyCounts(**b) for b in c["per_body"])
        c["per_lid_elements"] = tuple(c["per_lid_elements"])
        c["radiation_mode_ids"] = tuple(c["radiation_mode_ids"])
        return Counts(**c)

    def classify_points(self, points, tank=None) -> PointClasses:
        """The verdict of the field-point rules (reference/inputs-model-objects §3) on any points: per point a status,
        one of "fluid", "near_body" (valid, closer to a low-order panel's centroid than its
        diameter: a warning), "tank", "no_tank", "empty_tank", "above_tank_surface",
        "above_free_surface", "below_sea_bed", "outside_domain", "on_body", "inside_waterplane",
        "on_lid", "inside_body", "not_finite"; and the body, lid and tank of the status (``body``:
        of a point on, near or inside a body or inside its waterplane; ``lid``: of a point on a
        lid; of a point in a tank ``body`` and ``tank``, the zero-based tank of its body as
        ``results.bodies[b].tanks[t]`` numbers it; -1 none). ``tank`` None: points in the sea,
        global coordinates (as ``first_order.field_points``); a tank of the model
        (``m.bodies[b].tanks[t]``): points inside it, body coordinates (as its ``field_points``; a
        tank the solve does not take, disabled or of a disabled body: "no_tank")."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1 and pts.shape == (3,):
            pts = pts.reshape(1, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have the shape (N, 3), got {pts.shape}")
        pts = np.ascontiguousarray(pts)
        if tank is not None and not (isinstance(tank, ModelObject) and tank.object_type == "Tank"
                                     and tank._o.model is self._doc):  # fmt: skip
            raise TypeError("tank must be None (the sea) or a tank of this model")
        status, body, lid, tank_ = self._doc.classify_points(pts, None if tank is None else tank._o)
        return PointClasses(tuple(status), body, lid, tank_)

    def motion_equation(self) -> MotionEquation:
        """The rigid-body matrices of the motion equation as the engine uses them: inertia,
        stiffness (hydrostatic, weight and additional) and additional damping, 6N x 6N in body axes
        about each R; ``ready`` False (and the matrices None) when a body lacks a centre of
        gravity, a mass or inertia."""
        inertia, stiffness, damping, ready = self._doc.motion_equation()
        if not ready:
            return MotionEquation(None, None, None, False)
        return MotionEquation(inertia, stiffness, damping, True)  # read-only, as every answer

    def panels(self, role=None, include_reflections=True) -> Panels:
        """The elements as the engine uses them (after a trim), for display: of every body in
        order (the elements of its mesh with the lids placed with it, then its surfaces of other
        meshes: the dipoles, interior and pressure surfaces and free-surface plates, the control
        surface, a generated interior surface), then the elements of no body (body -1: the lids not
        placed with a body, the fixed dampers, the free surface). The seven key columns (``body``,
        ``role``, ``tank``, ``owner``, ``element``, ``part``, ``reflection``) identify a row
        (``Panels``). ``role``: None every role but "control_surface" and "free_surface" (the
        auxiliary surfaces), or a role name or a non-empty list of them, of "wetted", "interior_surface",
        "dipole", "dipole_damper", "pressure_surface", "lid", "tank", "fixed_damper",
        "control_surface", "free_surface" (the panelled zone of the potential QTF's free surface as
        the solve integrates it: the given mesh's panels with their reflections, or the generated
        panels; the cells have none, see ``free_surface_points``), "free_surface_plate";
        ``include_reflections`` also the copies about the symmetry planes. One body:
        ``body.panels``. The frequencies decide the auto sizes of the generated surfaces."""
        return _panels(self._doc.panels(_flag(include_reflections), _roles(role)))

    def free_surface_points(self, zone: str = "panelled") -> FreeSurfacePoints:
        """The points [n, 3], global, and the weights [n] of one zone of the potential QTF's free
        surface as the solve integrates it (second_order.free_surface_*): "panelled" (m^2: the
        panels' or cells' Gauss points), "quadrature" (m^2: the quadrature zone) or "outer_circle"
        (m: the line integral around the outer circle). The frequencies decide the auto sizes and
        the order. It applies the rules of the free surface (its items, the bodies' geometry and
        surfaces, the environment and its domain), not those of the frequencies, the orders or the
        other options; a zone whose auto values the frequencies decide is refused without a
        frequency between 0 and inf."""
        pts, w = self._doc.free_surface_points(str(zone))
        return FreeSurfacePoints(_frozen(pts), _frozen(w))

    def tessellation(self, role=None, divisions=8, include_reflections=True) -> Tessellation:
        """A quadrilateral mesh for display of the elements of ``panels`` with the same ``role``
        and ``include_reflections``, each quad with the seven keys of the panels row of its element:
        a low-order panel one quad, a patch ``divisions`` quads in u and v on its exact surface (an
        integer, or a pair (u, v), 1..256; "spans": one quad per span of the potential
        discretization of each patch)."""
        return _tessellation(self._doc.tessellation(_roles(role), _divisions(divisions), _flag(include_reflections)))

    def mode_displacements(self, points, bodies, elements) -> np.ndarray:
        """The displacement of every coordinate at points on the bodies, for animations:
        ``points`` [n, 3] global (e.g. the centroids of ``panels``), ``bodies`` and ``elements``
        [n] the body index and the one-based stored element of the body's mesh of each (as
        ``Panels.element``; it selects the pieces of the generalized modes; an element the body
        does not use is refused). Returns [n, D, 3] in global axes: the unit rigid modes (rotations
        about R, in body axes) and the generalized modes; zero for the pressure coordinates and
        other bodies. With the motions xi [D]: x(t) = x + Re(sum_j xi_j u_j e^(i w t))."""
        pts = np.ascontiguousarray(points, dtype=np.float64)
        b = np.ascontiguousarray(bodies, dtype=np.int32)
        r = np.ascontiguousarray(elements, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 3 or b.shape != (len(pts),) or r.shape != (len(pts),):
            raise ValueError("points must have the shape (n, 3), bodies and elements (n,)")
        return self._doc.mode_displacements(pts, b.tolist(), r.tolist())

    def mesh_quality(self) -> MeshQuality:
        """The quality of the meshes of every body (the same findings are issued as warnings when
        the model is checked or solved; one body: ``body.mesh_quality``, a mesh alone:
        ``mesh.quality``). The flags of an element are those of "orientation", "free_edge",
        "non_manifold", "coarse" that apply."""
        return _quality(self._doc.mesh_quality())

    def solve(self, progress=None):
        """Solves the model; its results are kept in it (``Model.results``) and returned.

        ``progress``: of one argument, ``progress(report)`` with a ``Progress`` for the stages (at
        most every 0.1 s) and for every finished frequency; of two, ``progress(done, total)``
        after each frequency. Returning ``False`` cancels the solve (``SolveCancelledError``); an
        exception raised in it cancels the solve and is raised here. The engine's warnings of the
        solve are issued as ``HydroBEMWarning``. The engine solves with the GIL released: other
        threads run meanwhile, and the callback (called on an engine thread) may read the model
        and cancel the solve (``Model.cancel`` from any thread)."""
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        res = Results(self._doc.solve(_reporter(progress)))
        _warn(res.warnings, 2)
        self._results = res
        return res

    def cancel(self) -> None:
        """Requests the cancellation of the running solve (from the progress callback or another
        thread), or of the next one when none runs; the solve raises SolveCancelledError."""
        self._doc.cancel()

    def progress(self) -> "SolveProgress":
        """The progress of the running solve, or of the last one, at this instant: from any thread
        while ``solve`` runs (it never waits for the solve nor slows it); before the first solve
        nothing to do."""
        (running, seconds, preparation, limits, frequencies, done, total, phase, workers,
         steps) = self._doc.progress()  # fmt: skip
        return SolveProgress(running, seconds, preparation,
                             tuple(LimitProgress(lim, FrequencyProgress(*p)) for lim, p in limits),
                             tuple(FrequencyProgress(*f) for f in frequencies), done, total, phase,
                             tuple(WorkerProgress(*w) for w in workers),
                             tuple(SecondOrderProgress(step, state, fraction,
                                                       tuple(FrequencyProgress(*i) for i in items))
                                   for step, state, fraction, items in steps))  # fmt: skip

    @property
    def state(self) -> str:
        """"reset" (no results), "calculating" or "calculated"."""
        return self._doc.state

    @property
    def results(self):
        """The results of the last solve (an ``hb.Results``, independent of later changes); a
        RuntimeError when the model is not calculated: not solved, or changed since."""
        state = self.state
        if state == "calculating":
            raise RuntimeError("the model is calculating: its results are there when the solve ends")
        if state != "calculated":
            self._results = None
            raise RuntimeError("the model has no results: it was not solved, or its inputs changed since")
        if self._results is None:  # a model loaded from a results file
            self._results = Results(self._doc.results())
        return self._results

    def reset(self):
        """Discards the results; the inputs stay."""
        self._results = None
        self._doc.reset()

    def clear(self):
        """Every input and the results back to a new model."""
        self._results = None
        self._doc.clear()

    def close(self):
        """Frees the model and its results now (else when the object goes); a RuntimeError while
        its solve runs (from the progress callback or another thread), which leaves both."""
        self._doc.close()
        self._results = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def save_results(self, path, overwrite: bool = False) -> None:
        """Writes the model with its results as a results file (.hbr)."""
        self._doc.save_results(Path(path), bool(overwrite))

    @classmethod
    def load_results(cls, path) -> Model:
        """A calculated model from a results file: its inputs and its results."""
        return cls(_core.Model.load_results(Path(path)))


# ------------------------------------------------------------------ the solve
@dataclasses.dataclass(frozen=True)
class Progress:
    """A progress report of a solve, the argument of a one-argument ``progress`` callback.

    Attributes
    ----------
    stage : {"statics", "assembly", "lu", "outputs", "frequency_done", "second_order", "phase", \
"worker_idle"}
        What the reporting engine thread is doing: the frequency-independent influence data,
        the matrices of a frequency, its LU, its outputs; or a frequency (or limit) finished; the
        parts of an item of a second-order step (``step``, ``item``); the phase of the solve
        changed (always reported); a worker has no item left in its phase.
    frequency : int or None
        Zero-based ordinary frequency of the reporting thread (-1 a limit); None for the statics,
        a second-order item and the reports of the solve as a whole.
    stage_done, stage_total : int
        Steps of the stage done and in all (rows, tiles, LU panels, output steps, the parts of a
        second-order item; for "frequency_done" the frequencies).
    frequencies_done, frequencies_total : int
        Frequencies and limits finished, and in all.
    seconds : float
        Since the start of the solve.
    worker : int or None
        The frequency worker that reports, 0 to frequency_threads - 1 (fixed for the whole solve);
        None for the work all threads share (the statics, a phase).
    phase : str
        The phase of the solve (as ``SolveProgress.phase``).
    step : str or None
        The second-order step of the report (as ``SecondOrderProgress.step``): its items' parts,
        or the stages of a frequency of a sub-problem of the potential QTF; None otherwise.
    item : int or None
        The item of that step (None: the work of the step all workers share).
    """

    stage: str
    frequency: int | None
    stage_done: int
    stage_total: int
    frequencies_done: int
    frequencies_total: int
    seconds: float
    worker: int | None = None
    phase: str = "preparation"
    step: str | None = None
    item: int | None = None


@dataclasses.dataclass(frozen=True)
class FrequencyProgress:
    """The progress of one frequency (or limit, or item of a second-order step) of a solve.

    Attributes
    ----------
    state : {"waiting", "running", "done", "failed", "cancelled"}
        Not begun, being solved, solved, its solve failed, or not solved (the solve was cancelled
        or stopped by the failure of another frequency).
    stage : {"assembly", "lu", "outputs", "second_order"} or None
        The stage it is in ("second_order" for an item of a second-order step that is not a
        frequency); None before its first step and once it has ended.
    fraction : float
        Its work done, 0 to 1: the stages weighted by their cost in this solve (measured by the
        frequencies done). It never decreases and is 1 once done.
    worker : int or None
        The frequency worker that runs it (or ran it), 0 to frequency_threads - 1; None while it
        waits, and for an item all the workers share.
    """

    state: str
    stage: str | None
    fraction: float
    worker: int | None = None


@dataclasses.dataclass(frozen=True)
class LimitProgress:
    """The progress of a limit of the radiation problem: ``limit`` "zero" (omega = 0) or
    "infinite", and its ``progress`` (a FrequencyProgress)."""

    limit: str
    progress: FrequencyProgress


@dataclasses.dataclass(frozen=True)
class WorkerProgress:
    """A frequency worker of a solve at one instant.

    Attributes
    ----------
    state : {"running", "idle", "done"}
        On an item (or on the work all threads share: the statics, the fits and the outer
        integral of the potential QTF); idle (no item left for it in the phase while other
        workers still run, and while the results are written); done (the solve has ended).
    task : str or None
        What it is on: "limit", "frequency", or the name of a second-order step; None without
        one.
    index : int or None
        The limit (in ``SolveProgress.limits``), the frequency or the item of the step; None for
        the work of a step all workers share.
    stage : str or None
        The stage of that item (as ``FrequencyProgress.stage``).
    fraction : float
        The fraction of that item done (0 without one).
    """

    state: str
    task: str | None
    index: int | None
    stage: str | None
    fraction: float


@dataclasses.dataclass(frozen=True)
class SecondOrderProgress:
    """A step of the second order of a solve.

    Attributes
    ----------
    step : str
        "quadratic_pressure_integration", "quadratic_control_surface" (their items the rows
        omega_a of the pairs), "potential_first_order" (the first-order sub-problem of the
        potential QTF, its items its frequencies), "potential_radiation" (the radiation at the
        pair frequencies, its items those), "potential_fits" (the far-field fits, one item),
        "potential_outer" (the outer integral, its items its batches of radii),
        "potential_pairs" (the inner free surface and the body of every pair, its items the pair
        frequencies), "potential_direct" (the direct method's solve, its items the pair
        frequencies).
    state : {"running", "done", "failed", "cancelled"}
    fraction : float
        Its work done, 0 to 1 (the mean of its items', a sub-problem's preparation as one more);
        never decreasing, 1 once done.
    items : tuple of FrequencyProgress
        Its items in their order.
    """

    step: str
    state: str
    fraction: float
    items: tuple


@dataclasses.dataclass(frozen=True)
class SolveProgress:
    """The state of the running solve, or of the last one, at one instant (``Model.progress``).

    Attributes
    ----------
    running : bool
        Whether the solve runs.
    seconds : float
        Since the solve began (until it ended).
    preparation : float
        The frequency-independent influence data done, 0 to 1.
    limits : tuple of LimitProgress
        The limits in the order they are solved (zero, then infinite); none without radiation or
        without the limits among the frequencies.
    frequencies : tuple of FrequencyProgress
        The ordinary frequencies in the order of the results (``Results.omegas``; the
        ``frequency`` of an error's location).
    done, total : int
        The frequencies and limits done, and in all.
    phase : str or None
        "preparation", "limits", "first_order", "second_order" (after every frequency is done),
        "finishing" (the results written), then "done", "cancelled" or "failed"; None before the
        first solve.
    workers : tuple of WorkerProgress
        The frequency workers (frequency_threads of them), each with its fixed number (its index).
    second_order : tuple of SecondOrderProgress
        The steps of the second order in the order they began (none before).
    """

    running: bool
    seconds: float
    preparation: float
    limits: tuple
    frequencies: tuple
    done: int
    total: int
    phase: str | None = None
    workers: tuple = ()
    second_order: tuple = ()


def _per_report(progress) -> bool:
    """True for a callback of one argument (every report), False for progress(done, total)."""
    try:
        params = [p for p in inspect.signature(progress).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]  # fmt: skip
    except (TypeError, ValueError):
        return False
    return len(params) == 1


def _reporter(progress):
    """The callback the engine calls with each report (a tuple of the fields of Progress): the
    user's ``progress`` of a ``Progress``, or of (done, total) after each frequency. What it
    returns is passed on: False cancels the solve."""
    if progress is None:
        return None
    if _per_report(progress):
        return lambda report: progress(Progress(*report))

    def frequencies(report):
        if report[0] == "frequency_done":
            return progress(report[4], report[5])
        return None

    return frequencies


def _warn(diags, stacklevel):
    for d in diags:
        if d.severity == "warning":
            warnings.warn(HydroBEMWarning(d.message, d.code, d.location), stacklevel=stacklevel + 1)


__all__ = ["BodyObject", "MeshList", "MeshObject", "Model", "ModelObject", "ObjectList", "Progress",
           "SecondOrderProgress", "WorkerProgress",
           "schema"]
