"""Regenerates the validation report: every figure into docs/validation/figures/ (the comparisons
and the meshes, mesh_<case>.png), the numbers the pages quote (docs/validation/figures/numbers.json),
one page per case (docs/validation/cases/<case>.md) and the index docs/validation/README.md.

    python validation/make_report.py [--quick | --replot | --pages] [group ...]

The groups (the scripts of validation/, all by default): hemisphere, cylinder, bottom_cylinder,
kernels, qtf, constants, two_truncated_cylinders, three_truncated_cylinders, two_submerged_spheres,
floating_owc_piston, floating_owc_chamber, cylinder_with_heave_plate, fish_tank_cylinders, dock,
floating_owc_cover. Each script computes its case (the low and, where the engine has it, the
higher order) and writes its figures and numbers; the pages are written from numbers.json after the
runs (a page whose numbers are missing is left as it is).
--quick runs coarser meshes and fewer values (a check of the scripts, not the report's numbers).
--replot redraws, without a solve, the figures whose numbers numbers.json holds (the groups
cylinder, qtf and constants: cylinder_convergence, qtf_kim_yue, qtf_convergence and both
constants figures); the other figures need their group run.
--pages only rewrites the pages and the index from numbers.json.
The engine must be built (scripts/build.bat Release or cargo build --release with the extension
module copied to python/hydrobem); the kernels group builds and runs a cargo example. The run
time of each group is printed and stored in numbers.json.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time

from common import CASES, NUMBERS, ROOT, record, reflow, table

GROUPS = ["hemisphere", "cylinder", "bottom_cylinder", "kernels", "qtf", "constants", "two_truncated_cylinders",
          "three_truncated_cylinders", "two_submerged_spheres", "floating_owc_piston", "floating_owc_chamber",
          "cylinder_with_heave_plate", "fish_tank_cylinders", "dock", "floating_owc_cover"]  # fmt: skip
# the order of the index: the cases against published and analytic solutions first, then the second
# order, the cases without an independent reference, and the method pages
ORDER = ["floating_hemisphere", "truncated_cylinder_finite_depth", "bottom_mounted_cylinder", "dock",
         "two_truncated_cylinders", "three_truncated_cylinders", "two_submerged_spheres", "cylinder_with_heave_plate",
         "fish_tank_cylinders", "floating_owc_chamber", "second_order_bottom_cylinder", "cylinder",
         "floating_owc_piston", "floating_owc_cover", "green_functions", "constants"]  # fmt: skip
# the methods of a case in the index, in words
METHODS = {"low, higher": "both methods", "low": "low order only", "kernel": "the kernel alone"}


def pages_of(module):
    """{page: (index items, page function, summary function, the groups of numbers.json it needs)}."""
    if hasattr(module, "PAGES"):
        return module.PAGES
    return {module.NAME: (module.INDEX, module.page, module.summary, getattr(module, "GROUPS", (module.NAME,)))}


def write_pages():
    """Writes the case pages whose numbers numbers.json holds, and the index."""
    numbers = json.loads(NUMBERS.read_text(encoding="utf-8"))
    rows = {}
    for group in GROUPS:
        for name, (index, page, summary, needs) in pages_of(importlib.import_module(group)).items():
            if not all(k in numbers for k in needs):
                print(f"  {name}: no numbers, page left as it is", flush=True)
                continue
            CASES.mkdir(parents=True, exist_ok=True)
            (CASES / f"{name}.md").write_bytes((reflow(page(numbers)) + "\n").encode("utf-8"))
            result = summary(numbers)
            rows[name] = [index["case"], f"{index['checked']} ({METHODS[index['methods']]})", index["sources"],
                          result[:1].upper() + result[1:], f"[page](cases/{name}.md)"]  # fmt: skip
    ordered = [rows[n] for n in ORDER if n in rows] + [r for n, r in rows.items() if n not in ORDER]
    run = numbers.get("run", {})
    seconds = run.get("seconds", {})
    total = seconds.get("total")
    timing = (f"\n\nThe whole report takes about {round(total / 60)} minutes on {run.get('threads')} threads "
              "(each group's time in `figures/numbers.json`, `run`)." if total else "")  # fmt: skip
    text = README.format(table=table(["Case", "What is checked", "Reference", "Result", "Page"], ordered),
                         timing=timing)  # fmt: skip
    (ROOT / "docs" / "validation" / "README.md").write_bytes((reflow(text) + "\n").encode("utf-8"))
    print(f"  pages: {len(rows)} written, index docs/validation/README.md", flush=True)


README = """
# Validation

Validation here means comparing HydroBEM's results with solutions that do not come from HydroBEM:
published tables and figures of the literature, closed-form analytic solutions, and semi-analytic
solutions (eigenfunction matching, multipoles, series) derived from published formulations and
implemented independently of the panel method in this repository. Where no such solution is known,
a case rests on the identities of the problem and on convergence, and its page says so.

Each case has a page of its own. It opens with the result in plain words, then describes the problem
and its meshes, the reference and its accuracy, the agreement, the convergence and any special topic
of the case. Both methods are compared wherever the engine has them: the low order (constant panels)
and the higher order (a B-spline potential on exact or spline geometry).

*Table 1. The cases: what each compares, against which reference, and the result.*

{table}

Unless a page says otherwise, a result is the largest difference over the compared range relative to
the largest reference value of each quantity. Values read from published figures are compared to the
accuracy of the reading, stated on each page.

How the comparisons are normalized and which references are used is described in
[method](method.md); what is not measured against an independent solution is listed in
[not verified](not_verified.md); the tests and their thresholds are in the [test plan](test-plan.md).

The examples of `examples/` show how to use HydroBEM and are a subset of these cases. The tests of
`tests/` are a fast subset with thresholds, run on every change; the pages hold the full comparison.
Every figure, number and page here is regenerated by the scripts of `validation/`
(`python validation/make_report.py`; `--quick` for a coarse check of the scripts, `--pages` to rewrite
the pages from `figures/numbers.json`).{timing}
"""


def main():
    quick = "--quick" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or GROUPS
    if "--pages" in sys.argv:
        write_pages()
        return
    if "--replot" in sys.argv:
        numbers = json.loads(NUMBERS.read_text(encoding="utf-8"))
        for name in names:
            module = importlib.import_module(name)
            if hasattr(module, "replot"):
                print(f"{name}: replot", flush=True)
                module.replot(numbers)
        return
    times = {}
    t0 = time.time()
    for name in names:
        print(f"{name}:", flush=True)
        t = time.time()
        importlib.import_module(name).main(quick)
        times[name] = round(time.time() - t, 1)
    times["total"] = round(time.time() - t0, 1)
    if names == GROUPS:
        record("run" + ("_quick" if quick else ""), {"seconds": times, "threads": os.cpu_count(),
                                                    "processor": platform.processor()})  # fmt: skip
    print("run time (s):", times)
    write_pages()


if __name__ == "__main__":
    main()
