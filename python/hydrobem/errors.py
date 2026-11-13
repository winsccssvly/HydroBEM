"""The exceptions and diagnostics of the engine (design.md, Errors): the extension module raises the
exception of the kind of an engine error (``exception_for``) with its location."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Location:
    """Where the engine found a diagnostic: the body, mesh, field point, lid and frequency
    (zero-based) and the mesh element (one-based) it applies to; None where it does not apply."""

    body: int | None = None
    mesh: int | None = None
    element: int | None = None
    field_point: int | None = None
    lid: int | None = None
    frequency: int | None = None


@dataclass(frozen=True)
class Diagnostic:
    """A warning of the engine, or an error recorded in a results file: "warning" or
    "error", its code (the warning's number), where it applies and its message."""

    severity: str
    code: int
    location: Location
    message: str


class UnsupportedProblemError(NotImplementedError):
    """A valid problem for which the engine has no solver route; the message
    names the combination (the supported ones: docs/reference/inputs-rules.md, section 2)."""


class SolveCancelledError(RuntimeError):
    """The solve was cancelled (the progress callback returned False)."""


class HydroBEMWarning(UserWarning):
    """A warning of the engine: valid input whose result may not be what is expected.
    ``code`` is the warning's number and ``location`` where it applies (as for errors)."""

    def __init__(self, message, code=0, location=None):
        super().__init__(message)
        self.code = code
        self.location = location or Location()


# the exception of each kind of engine error (the engine's ErrorKind)
_EXCEPTIONS = {
    "value": ValueError,
    "type": TypeError,
    "unsupported": UnsupportedProblemError,
    "state": RuntimeError,
    "cancelled": SolveCancelledError,
    "internal": RuntimeError,
    "argument": ValueError,
    "numerical": np.linalg.LinAlgError,
    "io": OSError,
    "exists": FileExistsError,
}


def exception_for(kind: str, message: str, location: Location | None = None) -> Exception:
    """The Python exception of an engine error of a kind ("value", "type", "unsupported", ...;
    design.md, Errors). ``location``: where the engine found it; set as the attribute
    ``location`` (all None when unknown)."""
    if kind == "exists" and message.endswith(" exists"):
        message += " (pass overwrite=True)"
    exc = _EXCEPTIONS.get(kind, RuntimeError)(message)
    exc.location = location or Location()
    return exc
