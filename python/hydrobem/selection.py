"""The element ranges of a mesh reference from a selection of elements: ``hb.ranges``."""

from __future__ import annotations

import numpy as np


def ranges(selection) -> list[tuple[int, int]]:
    """The element ranges of a mesh reference: the sorted disjoint inclusive ranges
    ``[(first, last), ...]`` of the elements selected, one-based as the elements are stored.

    ``selection`` is a boolean mask over the stored elements of a mesh (``mask[k]`` element k + 1),
    or the one-based numbers of elements (in any order; a number given twice counts once). With a
    mesh's elements (``mesh.elements()``) a selection by geometry is a mask::

        el = m.meshes["spar"].elements()
        top = (el.normals[:, 2] > 0.99) & (np.abs(el.centroids[:, 2]) < 1e-6)
        m.bodies["spar"].interior_surface = ("spar", hb.ranges(top))

    A ValueError when nothing is selected: a mesh reference without ranges is the whole mesh.
    """
    a = np.asarray(selection)
    if a.ndim != 1:
        raise ValueError(f"a selection is a mask or element numbers of one dimension, not of shape {a.shape}")
    if a.dtype == np.bool_:
        elements = np.flatnonzero(a) + 1
    elif a.size == 0 or np.issubdtype(a.dtype, np.integer):
        elements = np.unique(a.astype(np.int64))
        if elements.size and elements[0] < 1:
            raise ValueError(f"the elements are one-based: {int(elements[0])} is not an element")
    else:
        raise TypeError(f"a selection is a boolean mask or integer element numbers, not of {a.dtype}")
    if elements.size == 0:
        raise ValueError("no element is selected (a mesh reference without ranges is every element of the mesh)")
    # the breaks between consecutive elements start new ranges
    cut = np.flatnonzero(np.diff(elements) != 1)
    first = np.concatenate([elements[:1], elements[cut + 1]])
    last = np.concatenate([elements[cut], elements[-1:]])
    return [(int(f), int(e)) for f, e in zip(first, last, strict=True)]
