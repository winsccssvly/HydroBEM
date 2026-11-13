"""The build of the engine library and the table of its result quantities.

The build reports the package version, a commit, the compiler and the linear-algebra library it
is built with (faer and its version). Every result quantity has a name, a type,
axes and a unit: the names are those of the Python binding (the attributes of hb.Results), and the
type and the number of axes are those of the arrays a solve returns."""

import os

import numpy as np
import pytest

import hydrobem as hb
from hydrobem._cpu import has_avx2
from hydrobem.queries import quantity_code, quantity_names

from regression_problems import PROBLEMS


def test_build_info():
    b = hb.build_info()
    assert b.version == hb.__version__ and isinstance(b.api_version, int) and b.api_version >= 32
    assert b.commit and b.compiler and b.linear_algebra.startswith("faer ")
    assert b.build_type in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel") and isinstance(b.avx2, bool)
    # whether the build uses AVX2 is the build's; an AVX2 build loads only on a processor that reports
    # AVX2 or cannot tell (the check before the import of the extension module)
    if b.avx2 and not os.environ.get("HYDROBEM_SKIP_CPU_CHECK"):
        assert has_avx2() is not False


def test_quantity_names_and_units():
    q = hb.quantities()
    assert len(q) == len(quantity_names())
    for code, name in enumerate(quantity_names()):
        assert q[name].code == code
        assert q[name].unit and q[name].description and q[name].dims
    assert q["added_mass"].unit == "kg m^(r_i+r_j)" and q["headings"].unit == "deg"
    # a name and its code of the table; an unknown name is refused (Results.quantity takes the code
    # of its name from the table)
    assert quantity_names()[quantity_code("added_mass")] == "added_mass"
    with pytest.raises(ValueError, match="no result quantity 'mass_added'"):
        quantity_code("mass_added")


@pytest.mark.parametrize("name", ["lo_cylinder_drift", "lo_cylinder_field_body_points", "ho_moonpool_lid"])
def test_quantity_types_and_axes_match_the_results(name):
    p = PROBLEMS[name]()
    q = hb.quantities()
    r = p._doc.solve()  # the engine's results
    arrays = {quantity_names()[k]: r.quantity(k) for k in r.available()}
    assert arrays
    for k, a in arrays.items():
        info = q[k]
        assert np.dtype(info.dtype) == a.dtype, k
        assert len(info.dims) == a.ndim, k
