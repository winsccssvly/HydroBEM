"""The published reference values written into the examples (examples/*/run*.py, self-contained:
they import nothing of the repository and read no data outside their folder) agree with those of
the validation report (validation/*.py) and of tests/data, the one source of truth. Both kinds of
script are parsed, not run: the module-level literal assignments are read with ast.

Values without a counterpart in the report are not checked here: examples/cylinder_with_heave_plate/
run_model_scale.py (Han et al. 2024, Figs. 9 and 11, read), examples/floating_owc_chamber/
run_open_heave.py (Mavrakos & Konispoliatis 2012, Table 3) and the zeros of Newman (2021, Section 3)
in examples/fish_tank_cylinders."""

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES, VALIDATION, DATA = ROOT / "examples", ROOT / "validation", ROOT / "tests" / "data"


def constants(path):
    """{name: value} of the module-level assignments of literals (np.array(literal) as its literal)."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and ast.unparse(value.func) in ("np.array", "numpy.array") and value.args:
                value = value.args[0]
            try:
                out[node.targets[0].id] = ast.literal_eval(value)
            except ValueError:
                continue
    return out


def example(case, script="run.py"):
    return constants(EXAMPLES / case / script)


def report(module):
    return constants(VALIDATION / f"{module}.py")


def test_cylinder_with_heave_plate():
    v = report("cylinder_with_heave_plate")
    pairs = [(example("cylinder_with_heave_plate")["FIG_4"], v["FIG_4"]),
             (example("cylinder_with_heave_plate", "run_thin_plate.py")["FIG_6"], v["FIG_6"])]  # fmt: skip
    bodies = example("cylinder_with_heave_plate", "run_higher_order.py")
    # BODIES is a dict of dict(...) calls: its a33 read from the source
    tree = ast.parse((EXAMPLES / "cylinder_with_heave_plate" / "run_higher_order.py").read_text(encoding="utf-8"))
    assert "BODIES" not in bodies
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "dict":
            kw = {k.arg: k.value for k in node.keywords}
            a33 = ast.literal_eval(kw["a33"])
            pairs.append(({"a33": a33}, v["FIG_4"] if ast.literal_eval(kw["thick"]) else v["FIG_6"]))
    assert len(pairs) == 4
    for ex, ref in pairs:
        for q, values in ex.items():
            for nu, x in values.items():
                assert ref[q][nu] == x, (q, nu)


@pytest.mark.parametrize("script", ["run.py", "run_higher_order.py"])
def test_three_truncated_cylinders(script):
    ex = example("three_truncated_cylinders", script)
    (kb,) = ex["KB"]
    ref = report("three_truncated_cylinders")["FIG_5"]
    for q, x in ex["FIG_5"].items():
        assert ref[q][kb] == x, q


@pytest.mark.parametrize("script", ["run.py", "run_higher_order.py"])
def test_two_submerged_spheres(script):
    ex = example("two_submerged_spheres", script)
    (nu_a,) = ex["NU_A"]
    ref = report("two_submerged_spheres")["FIGS"]
    for q, x in ex["FIGS_2_10"].items():
        assert ref[q][nu_a] == x, q


@pytest.mark.parametrize("script", ["run.py", "run_higher_order.py"])
def test_two_truncated_cylinders(script):
    assert example("two_truncated_cylinders", script)["TABLE_2"] == report("two_truncated_cylinders")["TABLE_2"]


@pytest.mark.parametrize("script", ["run.py", "run_higher_order.py"])
def test_floating_owc_chamber(script):
    # PAPER: {ka: (Table 1, Table 2)} at g_T = 1, the second column of the report's tables, whose
    # rows are the report's KA_12
    ex = example("floating_owc_chamber", script)["PAPER"]
    v = report("floating_owc_chamber")
    for ka, (t1, t2) in ex.items():
        row = v["KA_12"].index(ka)
        assert (v["TABLE_1"][row][1], v["TABLE_2"][row][1]) == (t1, t2), ka


@pytest.mark.parametrize("script", ["run.py", "run_higher_order.py"])
def test_floating_hemisphere(script):
    ex = np.array(example("floating_hemisphere", script)["HULME"])
    full = np.loadtxt(DATA / "hulme1982_full.txt")
    for row in ex:
        assert np.array_equal(full[full[:, 0] == row[0]][0], row), row[0]


def test_second_order_bottom_cylinder():
    ex = example("second_order_bottom_cylinder")["KIM_YUE"]
    rows = [line.split() for line in (DATA / "kim_yue1990.txt").read_text().splitlines() if not line.startswith("#")]
    ref = {(k, float(a), float(b)): float(x) for k, a, b, x in rows}
    assert all(ref[key] == x for key, x in ex.items())
