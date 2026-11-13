"""Models of the tests: ``model`` puts bodies, lids, fixed dampers and meshes given as dicts of the
engine's items into an ``hb.Model``, with the defaults of the test helpers following the model
rules (a heading only with diffraction, a field output requested whenever field points are given).

A shorthand of the test dicts (not of the engine): the "mesh" of a surface may be a mesh dict, a
callable that makes a mesh, a mesh name, or ``(mesh, ranges)`` (bodies.part), which becomes the
items mesh and ranges; a generalized mode's piece takes ``(mesh, ranges)`` as its ranges; an
interior surface may be given as its method (a string) or as ``(mesh, ranges)`` (the method
"mesh")."""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
from scipy import special as sp

import hydrobem as hb

from bodies import TANK_POINTS, flat, named_tank_vessel, plate_column, surface
from meshes import box_panels

# the items of each object type that name a mesh, and the object lists of each type
_SCHEMA = hb.engine.schema()


def _mesh_refs(type_name):
    return [k for k, d in _SCHEMA[type_name]["items"].items()
            if d["kind"] == "reference" and d["target"] == "meshes"]  # fmt: skip


def _items_of(type_name):
    return set(_SCHEMA[type_name]["items"])


class _Meshes:
    """Adds the meshes of the dicts to a model: a dict of mesh items, a callable ``f(model, name)``
    that makes a mesh object (``gdf``, ``analytic``), or the name of a mesh of the model, alone or
    in ``(mesh, ranges)`` (the items mesh and ranges). The same dict or callable twice is one
    mesh."""

    def __init__(self, m):
        self.m, self.made = m, {}

    def name(self, spec, hint):
        if spec is None or isinstance(spec, str):
            return spec
        if id(spec) in self.made:
            return self.made[id(spec)][0]
        name, k = hint, 2
        while name in self.m.meshes:
            name, k = f"{hint}_{k}", k + 1
        if callable(spec):
            spec(self.m, name)
        else:
            add_mesh(self.m, name, spec)
        self.made[id(spec)] = (name, spec)  # keeps the spec alive while its id is a key
        return name

    def resolve(self, type_name, items, hint):
        """The own items with their meshes named (``(mesh, ranges)`` as the items mesh and
        ranges; the surfaces of a body given as a dict, a method or (mesh, ranges) as their flat
        items, ``flat_surface``); the objects of the lists are left in the dict, for ``add`` to add
        once their parent is made."""
        out = dict(items)
        if type_name == "Body":
            for k in ("interior_surface", "control_surface"):
                if k in out and not isinstance(out[k], str):
                    out.update(flat_surface(k, out.pop(k)))
        for k in _mesh_refs(type_name):
            if k in out:
                v = out[k]
                if isinstance(v, tuple):
                    out[k], out[k[: -len("mesh")] + "ranges"] = v[0], _ranges(v[1])
                # a surface's mesh named after its surface, as when it was an object of its own
                surface = k[: -len("_mesh")] if k.endswith("_mesh") else k
                out[k] = self.name(out[k], hint if k == "mesh" or hint == surface else f"{hint}_{surface}")
        if type_name == "ModePiece" and isinstance(out.get("mesh"), tuple):
            out["ranges"] = _ranges(out.pop("mesh")[1])  # a piece takes the elements of the body's mesh
        return out

    def add(self, lst, type_name, d, hint):
        """Adds the object of a dict to a list (``ObjectList``): its own items and surfaces by
        ``add``, then the objects of its lists, each added to the new object's list."""
        d = dict(d)
        lists = _SCHEMA[type_name]["lists"]
        kids = {k: d.pop(k) for k in list(d) if k in lists}
        items = self.resolve(type_name, d, hint)
        t, n = items.pop("type", None), items.pop("name", None)
        obj = lst.add(*([] if n is None else [n]), **({"type": t} if t else {}), **items)
        for k, v in kids.items():
            t = lists[k]["types"][0]
            for i, c in enumerate(v):
                self.add(getattr(obj, k), c.get("type", t), c, f"{hint}_{k}{i + 1}")
        return obj


_FREE_SURFACE = ("panel_size", "inner_radius", "mesh", "ranges", "quadrature_points")


def flat_surface(kind: str, v) -> dict:
    """The flat items of a surface given as a dict of its method and items (the shorthand of the
    tests): ``kind`` "interior_surface" or "control_surface" (a body's: ``{"method": "cylinder",
    "radius": 2.0}`` as ``control_surface="cylinder", control_surface_radius=2.0``; a method alone,
    (mesh, ranges) the method "mesh", None the method "none") or "free_surface" (second_order's:
    ``method`` as ``free_surface_method``, the items of the panelled zone prefixed
    ``free_surface_``, the quadrature items as they are)."""
    if v is None:
        v = {"method": "none"}
    elif isinstance(v, str):
        v = {"method": v}
    elif isinstance(v, tuple):
        v = {"method": "mesh", "mesh": v}
    v = dict(v)
    if kind == "free_surface":
        out = {"free_surface_method": v.pop("method")} if "method" in v else {}
        out.update({f"free_surface_{k}" if k in _FREE_SURFACE else k: x for k, x in v.items()})
        return out
    out = {kind: v.pop("method", "detect" if kind == "interior_surface" else "none")}
    out.update({f"{kind}_{k}": x for k, x in v.items()})
    return out


def _ranges(r):
    """Ranges of elements: one (first, last) or a sequence of them, as a list of pairs."""
    r = list(r)
    if len(r) == 2 and all(isinstance(x, (int, np.integer)) for x in r):
        return [tuple(r)]
    return [tuple(x) for x in r]


def add_mesh(m, name, spec: dict):
    """Adds a mesh dict to a model: its items by ``add``, then its patches to its list (a list's
    objects are added once their parent is made); the mesh object."""
    items = dict(spec)
    patches = items.pop("patches", ())
    mesh = m.meshes.add(name, **items)
    for c in patches:
        c = dict(c)
        mesh.patches.add(c.pop("name", None), type=c.pop("type", None), **c)
    return mesh


def gdf(path, **kw):
    """A mesh read from a GDF file (and SPL file) by the engine (``MeshList.load``)."""
    return lambda m, name: m.meshes.load(name, path, **kw)


def analytic(shape, rows, **kw):
    """The named analytic shape with its dimensions and options as keywords (``MeshList.analytic``)."""
    return lambda m, name: m.meshes.analytic(name, shape, rows, **kw)


def add_body(m, body: dict, meshes=None):
    """Adds a body dict (its meshes and the objects of its lists too) to a model; the body object."""
    body = {"name": "body", **body}
    return (meshes or _Meshes(m)).add(m.bodies, "Body", body, body["name"])


def load_script(name: str):
    """The script scripts/<name>.py as a module (its main not run): the tests of the generators."""
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def threads_per_test() -> int:
    """The engine threads of one test: the processor's threads shared by the pytest workers
    (pytest-xdist's PYTEST_XDIST_WORKER_COUNT; one worker without it)."""
    return max(1, (os.cpu_count() or 1) // int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))


def model(bodies=(), omegas=(), headings=None, *, environment=None, options=None, outputs=None, field_points=None,
          lids=(), dampers=(), qtf_free_surface=None) -> hb.Model:  # fmt: skip
    """An hb.Model of body dicts (one or a list; "name" defaults to "body", the meshes as
    ``_Meshes`` takes them), the lid and fixed damper dicts, and dicts of the environment, options
    and outputs items; the items of ``options`` and ``outputs`` that belong to the first or second
    order or the environment are set there (the tests give the method and the outputs together),
    ``qtf_free_surface`` the items of the potential QTF's free surface as a dict (``flat_surface``).
    One heading (0 degrees) when
    diffraction is solved and none otherwise; ``field_pressure`` switched on when field points (in
    the sea) are given without a field output."""
    options = {**(options or {}), **(outputs or {})}
    if "frequency_threads" not in options and "inner_threads" not in options:
        options["frequency_threads"] = threads_per_test()  # the results do not depend on it
    first = {k: options.pop(k) for k in list(options) if k in _items_of("FirstOrder")}
    second = {k: options.pop(k) for k in list(options) if k in _items_of("SecondOrder")}
    environment = {**{k: options.pop(k) for k in list(options) if k in _items_of("Environment")},
                   **(environment or {})}  # fmt: skip
    if headings is None:
        headings = (0.0,) if first.get("diffraction", True) else ()
    if field_points is not None and not (first.get("field_pressure") or first.get("field_velocity")):
        first["field_pressure"] = True
    m = hb.Model()
    meshes = _Meshes(m)
    bodies = [bodies] if isinstance(bodies, dict) else list(bodies)
    # the couplings once every body is there (a coupling names another body)
    couplings = [b.get("couplings") for b in bodies]
    for b in bodies:
        add_body(m, {k: v for k, v in b.items() if k != "couplings"}, meshes)
    for b, c in zip(m.bodies, couplings, strict=True):
        for x in c or ():
            b.couplings.add(**x)
    for lid in [lids] if isinstance(lids, dict) else lids:
        lid = dict(lid)
        name = lid.pop("name")
        m.lids.add(name, **meshes.resolve("DampingLid", lid, f"lid_{name}"))
    for k, d in enumerate([dampers] if isinstance(dampers, dict) else dampers):
        m.fixed_dampers.add(**meshes.resolve("FixedDamper", d, f"damper{k + 1}"))
    m.environment.omegas = np.asarray(omegas, dtype=np.float64)
    m.environment.headings = np.asarray(headings, dtype=np.float64)
    m.environment.update(**environment)
    m.options.update(**options)
    m.first_order.update(**first)
    m.second_order.update(**second)
    if qtf_free_surface is not None:
        m.second_order.update(**meshes.resolve("SecondOrder", flat_surface("free_surface", qtf_free_surface),
                                               "free_surface"))
    if field_points is not None:
        m.first_order.field_points = np.asarray(field_points, dtype=np.float64)
    return m


def two_tank_vessels(a=None, b=None, **outputs) -> hb.Model:
    """Two barges with a tank each (bodies.named_tank_vessel "a" at x = 0 and "b" at x = 6, with
    the items ``a`` and ``b``, e.g. a coupling of a to b), the field points bodies.TANK_POINTS: one
    in the tank of a, one in that of b (body axes), one in the sea; one frequency (1.2 rad/s) and
    heading; the field pressure and the motion besides ``outputs``. The field points of the solve
    (the sea's, then the tanks') are TANK_POINTS[2], [0], [1] in global axes."""
    bodies = [named_tank_vessel("a", 0.0, **(a or {})), named_tank_vessel("b", 6.0, **(b or {}))]
    for body, k, x in ((bodies[0], 0, 0.0), (bodies[1], 1, 6.0)):
        body["tanks"] = [{**body["tanks"][0], "field_points": TANK_POINTS[[k]] - [x, 0.0, 0.0]}]
    return model(bodies, [1.2], [0.0], field_points=TANK_POINTS[[2]],
                 outputs={"field_pressure": True, "motion": True, **outputs})  # fmt: skip


RHO, G = 1000.0, 9.80665


def solve(bodies, omegas, headings=None, depth=math.inf, progress=None, environment=None, **kw):
    """The results of ``model`` in water of RHO and G."""
    env = {"depth": depth, "density": RHO, "gravity": G, **(environment or {})}
    return model(bodies, omegas, headings, environment=env, **kw).solve(progress)


# the path in the result tree of each quantity with a view ("excitation_diffraction":
# "excitation.diffraction")
_PATHS = {info.code: info for info in hb.quantities().values()}
_VIEWS = {_PATHS[v["code"]].name: v["path"] for v in hb._core.result_views()}


def view(r, name: str):
    """The view of a quantity of results or of a body's results (``BodyResults``) by its quantity
    name (``hb.quantities()``) or its path in the result tree: ``view(r, "excitation_diffraction")``
    is ``r.excitation.diffraction``; None when not computed."""
    out = r
    for part in _VIEWS.get(name, name).split("."):
        out = getattr(out, part)
    return out


def peak(a):
    """The largest modulus of an array."""
    return np.abs(a).max()


def hulme(ka=(0.5, 1.0, 2.0)) -> np.ndarray:
    """The rows (Ka, A0, B0, A1, B1) of Hulme (1982) tables 1 and 2 at the given Ka
    (tests/data/hulme1982_full.txt, scripts/extract_hulme.py), in that order."""
    ref = np.loadtxt(Path(__file__).resolve().parents[1] / "data" / "hulme1982_full.txt")
    rows = [np.flatnonzero(np.isclose(ref[:, 0], k, rtol=0, atol=1e-9)) for k in ka]
    assert all(len(r) == 1 for r in rows), ka
    return ref[[r[0] for r in rows]]


def close(a, b, tol=1e-6):
    """Asserts that a equals b to tol of the peak of b (the largest difference, any entry)."""
    assert peak(a - b) <= tol * peak(b), (peak(a - b) / peak(b), tol)


def hydrostatics(m: hb.Model) -> list[hb.Hydrostatics]:
    """Hydrostatics of every body (the query of each body, theory §3.2)."""
    return [b.hydrostatics() for b in m.bodies]


def interior_vertices(m: hb.Model, body=0):
    """The interior free surface of a body as the engine uses it (given or generated): the
    vertices [n, 4, 3] of its elements of the role "interior", global, expanded."""
    return m.bodies[body].panels(role="interior_surface").vertices


def endpoints(r: hb.Results) -> dict:
    """The added mass at the limits that were computed, by "zero" and "infinite"."""
    out = {k: r.quantity(f"added_mass_{k}_frequency") for k in ("zero", "infinite")}
    return {k: v for k, v in out.items() if v is not None}


def drift_constants(omega, gravity, depth):
    """k, C (the propagating part of G is -i C Z(z) Z(zeta) H0(kR)) and K_e = 2 k^2 int Z^2 dz
    (theory §11.1): the reference of the drift tests."""
    K = omega * omega / gravity
    k = hb.wavenumber(omega, depth, gravity)
    if not math.isfinite(depth):
        return k, 2.0 * math.pi * K, K
    x = k * depth
    A0 = 2.0 * math.pi * k / (-math.expm1(-4.0 * x) + 4.0 * x * math.exp(-2.0 * x))
    C = A0 * (1.0 + math.exp(-2.0 * x)) ** 2
    Ke = K + k * k * depth / math.cosh(x) ** 2
    return k, C, Ke


def bottom_cylinder(ka, depth, a=1.0):
    """|X_1| / (rho g A a^2) and F_x / (rho g A^2 a) of the fixed vertical cylinder of radius a
    standing on the sea bed in the depth h, in waves along +x: the surge exciting force of MacCamy &
    Fuchs (1954), Beach Erosion Board TM 69, and the mean drift force by pressure integration of the
    same scattering solution (Havelock 1940, Proc. R. Soc. A 175, 409-421, eq. 36 in deep water).

    Time factor exp(i w t), k tanh kh = nu, Z = cosh k(z + h) / cosh kh, H_m = H_m^(2): on the wall
    the total potential is (g A / w) Z(z) S(theta), S = sum_m eps_m (-i)^m (-2i) cos(m theta) /
    (pi ka H_m'(ka)) (the Wronskian of J_m and Y_m); |X_1| = 4 rho g A tanh(kh) / (k^2 |H_1'(ka)|);
    F_x = (rho/4) int |grad phi|^2 n_x dS - (rho g/4) oint |zeta|^2 n_x dl (n out of the body), with
    int Z^2 dz = (h/2 + sinh 2kh/(4k)) / cosh^2 kh and int Z_z^2 dz = k^2 (sinh 2kh/(4k) - h/2) /
    cosh^2 kh. The series to the order ka + 40, the theta integrals by the trapezoidal rule of the
    periodic integrand (4096 points): both to the double round-off."""
    k = ka / a
    nu = k * math.tanh(k * depth)

    def dh2(m, x):  # H_m^(2)'(x)
        return sp.jvp(m, x) - 1j * sp.yvp(m, x)

    theta = np.linspace(0.0, 2.0 * np.pi, 4097)[:-1]
    s, ds = np.zeros_like(theta, dtype=complex), np.zeros_like(theta, dtype=complex)
    for m in range(int(ka + 40) + 1):
        c = (1.0 if m == 0 else 2.0) * (-1j) ** m * (-2j) / (math.pi * ka * dh2(m, ka))
        s += c * np.cos(m * theta)
        ds -= m * c * np.sin(m * theta)
    force = 4.0 * math.tanh(k * depth) / (k * k * abs(dh2(1, ka))) / a**2
    c2 = math.cosh(k * depth) ** 2
    z2 = (depth / 2 + math.sinh(2 * k * depth) / (4 * k)) / c2
    zz2 = k * k * (math.sinh(2 * k * depth) / (4 * k) - depth / 2) / c2
    # (g A / w)^2 = g A^2 / nu; dS = a dtheta dz, dl = a dtheta, n_x = cos theta
    wall = (a / 4) / nu * 2 * np.pi * ((z2 * abs(ds) ** 2 / a**2 + zz2 * abs(s) ** 2) * np.cos(theta)).mean()
    line = -(a / 4) * 2 * np.pi * (abs(s) ** 2 * np.cos(theta)).mean()
    return force, (wall + line) / a


def circular_tank(depth, K, n_terms=200):
    """A11, A15, A55 per unit density [3, F] of the fluid in a vertical circular tank of radius 1
    and fluid depth d, forced in surge and in pitch about the centre of its free surface (z = 0) at
    the wavenumbers K = w^2 / g: the closed-form series of Faltinsen & Timokha (2009), Sloshing, CUP,
    §5.4.4, for the potentials per unit velocity, the normal into the tank fluid, A_ij = -rho int n_i
    phi_j over the wall and bottom (theory §2.1). With k_n the zeros of J1', r = sum a_n J1(k_n r) /
    J1(k_n) on 0 <= r <= 1, a_n = 2 / (k_n^2 - 1), and C, S, T the cosh, sinh, tanh of k_n d:
    - surge: phi1 = [r + sum c_n J1(k_n r) / J1(k_n) cosh k_n(z + d)] cos(theta), c_n = K a_n /
      (k_n S - K C) from K phi = phi_z on z = 0: A11 = pi (d + sum c_n S / k_n);
    - pitch: phi5 = [r z + sum J1(k_n r) / J1(k_n) (alpha_n cosh k_n(z + d) + beta_n sinh k_n(z + d))]
      cos(theta), beta_n = -2 a_n / k_n from phi_z = -r on the bottom, alpha_n = [a_n + beta_n (k_n C -
      K S)] / (K C - k_n S): A55 = pi [d^3/3 - d/4 + sum ((alpha_n (2 - C) - beta_n S) / k_n^2 +
      beta_n d / k_n)], A15 = pi [-d^2/2 + sum (alpha_n S + beta_n (C - 1)) / k_n].
    Written with exp(-k_n d) so that nothing overflows; the terms decay like k_n^-3 or faster: 200
    terms leave < 1e-6."""
    kn = sp.jnp_zeros(1, n_terms)[:, None]
    K = np.atleast_1d(K)[None, :]
    e = np.exp(-kn * depth)
    t = np.tanh(kn * depth)
    sech = 2.0 * e / (1.0 + e * e)
    a = 2.0 / (kn**2 - 1.0)
    q = K * a / (kn * t - K)  # c_n C
    a11 = np.pi * (depth + np.sum(q * t / kn, axis=0))
    beta = -2.0 * a / kn
    delta_c = (a + beta * (kn + K) * e) / (K - kn * t)  # (alpha_n + beta_n) C
    a55 = np.pi * (depth**3 / 3 - depth / 4
                   + np.sum((-2.0 * beta + beta * e + 2.0 * delta_c * sech - delta_c) / kn**2 + beta * depth / kn,
                            axis=0))  # fmt: skip
    a15 = np.pi * (-(depth**2) / 2 + np.sum((beta * e - beta + t * delta_c) / kn, axis=0))
    return np.array([a11, a15, a55])


def _generalized():
    heave = {"name": "g-heave", "pieces": [{"w": [[1.0, 0, 0, 0]]}], "parity": "heave"}
    box = surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True), symmetry="xz_and_yz")
    every = [(1, len(box["vertices"]))]  # the ranges of every element
    bend = {"name": "bend", "parity": "heave",
            "pieces": [{"w": [[0.4, 2, 0, 0], [-0.5, 0, 0, 0]], "ranges": every}, {"w": [[0.1, 0, 2, 0]]}]}
    lo = {"name": "box", "mesh": box, "generalized_modes": [heave, bend], "mass": "displaced",
          "center_of_gravity": [0.0, 0.0, -0.1],
          "radii_of_gyration": [0.3, 0.4, 0.5], "additional_stiffness": np.eye(6) * 10.0}  # fmt: skip
    return model(lo, [1.5, 2.5], [0.0, 30.0], options={"motion": True, "frequency_threads": 2})


def _dampers():
    # the sheets of the plate column as a damper of the water (at rest)
    plate = plate_column()
    damper = {"mesh": plate.pop("dipoles")[0]["mesh"], "damping_factor": 0.5}
    return model(plate, [1.5], [0.0], environment={"depth": 4.0}, options={"higher_order": True}, dampers=[damper])


def _analytic():
    cyl = analytic("circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0, draft=0.5)
    return model({"mesh": cyl}, [0.0, 2.0, math.inf], [0.0], options={"higher_order": True})


def _trimmed():
    # the hull reaches above the water: the body is trimmed (and cut at z = 0) at the build
    q = box_panels(1.0, 0.5, 1.0, n=(2, 1, 2), quarter=True) + [0.0, 0.0, 0.25]
    hull = flat(q, (2, 2, 3, 3, 3, 4), symmetry="xz_and_yz")
    return model({"mesh": hull, "modes": ["heave", "pitch"], "trim": [0.0, 3.0, 0.0]}, [2.0], [0.0],
                 options={"higher_order": True})  # fmt: skip


# models with the objects the regression problems do not have, by name: each call makes a new
# model, so that no test shares the state of another
FEATURES = {"generalized": _generalized, "dampers": _dampers, "analytic": _analytic, "trimmed": _trimmed}


def written(stem) -> list[Path]:
    """The files of a stem (``<stem>.<ext>``) in its folder: what a writer made (the writers
    return no file names)."""
    stem = Path(stem)
    return sorted(f for f in stem.parent.glob(stem.name + ".*") if f.is_file())


# the six rigid modes about the origin as generalized modes of the same displacement (theory
# §10.1): translations e_j and rotations e_j x x, each with the parity of its rigid mode
RIGID_AS_GENERALIZED = [
    {"name": "g-surge", "pieces": [{"u": [[1.0, 0, 0, 0]]}], "parity": "surge"},
    {"name": "g-sway", "pieces": [{"v": [[1.0, 0, 0, 0]]}], "parity": "sway"},
    {"name": "g-heave", "pieces": [{"w": [[1.0, 0, 0, 0]]}], "parity": "heave"},
    {"name": "g-roll", "pieces": [{"v": [[-1.0, 0, 0, 1]], "w": [[1.0, 0, 1, 0]]}], "parity": "roll"},
    {"name": "g-pitch", "pieces": [{"u": [[1.0, 0, 0, 1]], "w": [[-1.0, 1, 0, 0]]}], "parity": "pitch"},
    {"name": "g-yaw", "pieces": [{"u": [[-1.0, 0, 1, 0]], "v": [[1.0, 1, 0, 0]]}], "parity": "yaw"},
]


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rigid_modes_as_generalized(body: dict, which, environment=None, fixed=False) -> dict:
    """The body dict (R at the origin, a mass, G and the diagonal inertia about G) with its rigid
    modes ``which`` (zero-based) as generalized modes of the same displacement, the other rigid
    modes kept free: mode_mass and mode_stiffness such that its equations of motion are the rigid
    body's, its mass about R (theory §3.1) and its restoring with the weight terms, less the
    restoring the engine computes for the generalized rows (theory §10.1: c_ij = rho g int n_j w_i,
    which for a rotation row holds the moment of the buoyancy of a translation, -rho g V, that the
    rigid rows about R do not have). ``fixed``: no rigid mode free and every generalized coordinate
    held by a stiffness of 1e12 instead (a generalized mode is always free)."""
    h = model(body, [1.0], environment=environment).bodies[0].hydrostatics()
    c = np.asarray(h.stiffness)
    names = ("surge", "sway", "heave", "roll", "pitch", "yaw")
    trial = {**body, "modes": [names[k] for k in range(6) if k not in which],
             "generalized_modes": [RIGID_AS_GENERALIZED[k] for k in which]}  # fmt: skip
    cg = model(trial, [1.0], environment=environment, outputs={"diffraction": False}).solve().mode_restoring
    m, g = body["mass"], np.asarray(body["center_of_gravity"], dtype=float)
    mr = np.zeros((6, 6))
    mr[:3, :3] = m * np.eye(3)
    mr[:3, 3:], mr[3:, :3] = -m * _skew(g), m * _skew(g)
    mr[3:, 3:] = np.diag([body["Ixx"], body["Iyy"], body["Izz"]]) + m * (g @ g * np.eye(3) - np.outer(g, g))
    n = 6 + len(which)
    q = list(range(6)) + list(which)  # the rigid mode of each coordinate
    mm, kk = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= 6 or j >= 6:
                mm[i, j] = mr[q[i], q[j]]
                kk[i, j] = c[q[i], q[j]] - cg[i, j]
    if fixed:
        kk = np.zeros((n, n))
        kk[6:, 6:] = 1e12 * np.eye(n - 6)
    keep = [] if fixed else [names[k] for k in range(6) if k not in which]
    return {**body, "modes": keep, "generalized_modes": [RIGID_AS_GENERALIZED[k] for k in which], "mode_mass": mm,
            "mode_stiffness": kk}  # fmt: skip
