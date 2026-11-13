"""HydroBEM: frequency-domain panel-method (BEM) hydrodynamics.

The engine is the Rust library hydrobem_core, bound as the extension module ``hydrobem._core``;
``hb.Model`` is its model. A typical solve::

    import hydrobem as hb

    m = hb.Model()
    m.meshes.load("cylinder", "cylinder.gdf")
    m.bodies.add("cylinder", mesh="cylinder", mass="displaced", center_of_gravity=[0, 0, 0],
                 radii_of_gyration=[1, 1, 1])
    m.environment.omegas = [0.5, 1.0, 2.0]
    m.environment.headings = [0.0]
    m.first_order.motion = True
    results = m.solve()
    results.write_wamit("out/cylinder")

A model file holds a model: ``m.save("case.hbm")``, ``hb.Model.load("case.hbm")``;
``m.save_results("run.hbr")`` adds the results.
"""

__version__ = "0.3.0"

from . import _cpu, engine  # noqa: F401 (_cpu: the processor checked before the extension module loads)
from .engine import (
    FrequencyProgress,
    LimitProgress,
    Model,
    Progress,
    SecondOrderProgress,
    SolveProgress,
    WorkerProgress,
)
from .errors import Diagnostic, HydroBEMWarning, Location, SolveCancelledError, UnsupportedProblemError
from .queries import (
    BodyCounts,
    BuildInfo,
    Counts,
    FreeSurfacePoints,
    Hydrostatics,
    ItemInfo,
    MemoryEstimate,
    MeshEdges,
    MeshElements,
    MeshGeometry,
    MeshQuality,
    Meta,
    MotionEquation,
    Panels,
    PatchPoints,
    PointClasses,
    QuantityInfo,
    Tessellation,
    build_info,
    quantities,
    wavenumber,
)
from .results import (
    BodyPointResults,
    BodyResults,
    Coordinate,
    ExcitationResults,
    FieldPointResults,
    LidResults,
    PressureSurfaceResults,
    Results,
    TankResults,
    load_results,
)
from .selection import ranges

__all__ = [
    "BodyCounts",
    "BodyPointResults",
    "BodyResults",
    "BuildInfo",
    "Coordinate",
    "Counts",
    "Diagnostic",
    "ExcitationResults",
    "FieldPointResults",
    "FreeSurfacePoints",
    "FrequencyProgress",
    "HydroBEMWarning",
    "Hydrostatics",
    "ItemInfo",
    "LidResults",
    "LimitProgress",
    "Location",
    "MemoryEstimate",
    "Meta",
    "MeshEdges",
    "MeshElements",
    "MeshGeometry",
    "MeshQuality",
    "Model",
    "MotionEquation",
    "Panels",
    "PatchPoints",
    "PointClasses",
    "PressureSurfaceResults",
    "Progress",
    "QuantityInfo",
    "Results",
    "SecondOrderProgress",
    "SolveCancelledError",
    "SolveProgress",
    "TankResults",
    "Tessellation",
    "UnsupportedProblemError",
    "WorkerProgress",
    "build_info",
    "engine",
    "load_results",
    "quantities",
    "ranges",
    "wavenumber",
]
