"""The results of a solve (reference/results): the engine's results (``hydrobem._core.Results``)
as the views of the result tree, by group and by owner. The classes are generated from the engine's
result tree (``_result_views``) on their hand-written parts (``_result_extras``); the engine
normalizes the quantities and writes the numeric files."""

from __future__ import annotations

from ._result_extras import Coordinate
from ._result_views import (
    BodyPointResults,
    BodyResults,
    ExcitationResults,
    FieldPointResults,
    LidResults,
    PressureSurfaceResults,
    Results,
    TankResults,
)


def load_results(path) -> Results:
    """The Results of a results file (.hbr)."""
    from .engine import Model

    return Model.load_results(path).results


__all__ = ["BodyPointResults", "BodyResults", "Coordinate", "ExcitationResults", "FieldPointResults", "LidResults",
           "PressureSurfaceResults", "Results", "TankResults", "load_results"]
