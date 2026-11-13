"""Error maps of the free-surface Green functions (docs/validation/cases/green_functions.md): the engine's infinite- and
finite-depth kernels at every point of the high-precision references of test-plan A2
(tests/data/green_inf_reference.txt and green_fin_reference.txt, written by
scripts/gen_green_reference_data.py and gen_green_finite_reference_data.py from the integral
representations at 30 digits), evaluated by the example crates/hydrobem/examples/green_error_map.rs
(built and run with cargo).

    python validation/kernels.py
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from common import ROOT, e, plt, record, save, tex

TARGET = 2e-6  # theory §8: the accuracy the tables are built to


def errors(which):
    cmd = ["cargo", "run", "--quiet", "--release", "-p", "hydrobem_core", "--example", "green_error_map", "--", which]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return np.array([[float(x) for x in s.split()] for s in out.stdout.splitlines() if s and not s.startswith("#")])


def main(quick=False):
    inf = errors("infinite")
    fin = errors("finite")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    e = np.maximum(inf[:, 2:4].max(axis=1), 1e-17)
    sc = ax.scatter(np.maximum(inf[:, 0], 1e-3), inf[:, 1], c=np.log10(e), s=9, cmap="viridis", vmin=-16, vmax=-5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("X = K R (X = 0 drawn at 10⁻³)")
    ax.set_ylabel("Y = −K (z + ζ)")
    ax.set_title(f"infinite depth: {len(inf)} points, largest {e.max():.1e}")
    fig.colorbar(sc, ax=ax, label="log₁₀ error of F and F_X")
    ax = axes[1]
    ef = np.maximum(fin[:, 5:9].max(axis=1), 1e-17)
    ordinary = fin[:, 4] == 0
    sc = ax.scatter(np.maximum(fin[ordinary, 0], 1e-15), np.maximum(fin[ordinary, 1], 1e-5), c=np.log10(ef[ordinary]),
                    s=9, cmap="viridis", vmin=-16, vmax=-5)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K h")
    ax.set_ylabel("R/h (R = 0 drawn at 10⁻⁵)")
    ax.set_title(f"finite depth: {ordinary.sum()} points (+{(~ordinary).sum()} at the limits), largest {ef.max():.1e}",
                 fontsize=8)  # fmt: skip
    fig.colorbar(sc, ax=ax, label="log₁₀ error (values and derivatives)")
    save(fig, "green_error_map")
    out = {"infinite_points": len(inf), "infinite_max": float(e.max()), "infinite_median": float(np.median(e)),
           "finite_points": len(fin), "finite_max": float(ef.max()), "finite_median": float(np.median(ef)),
           "finite_limits_max": float(ef[~ordinary].max()), "target": TARGET}  # fmt: skip
    print("  kernels:", out)
    record("green_error_map", out)


NAME = "green_functions"
GROUPS = ("green_error_map",)
INDEX = {"case": "Green functions", "checked": "infinite- and finite-depth free-surface Green functions and "
         "derivatives", "sources": "integral representations at 30 digits (mpmath)", "methods": "kernel"}  # fmt: skip


def summary(n):
    d = n["green_error_map"]
    verdict = "within the target" if max(d["infinite_max"], d["finite_max"]) <= d["target"] else "above the target"
    return (f"{verdict} {tex(d['target'], 0)} everywhere: largest error {tex(d['infinite_max'])} (infinite "
            f"depth) and {tex(d['finite_max'])} (finite depth)")


def page(n):
    d = n["green_error_map"]
    worst = max(d["infinite_max"], d["finite_max"], d["finite_limits_max"])
    verdict = ("within the target" if worst <= d["target"] else "not everywhere within the target")
    return rf"""
# Green functions

**Result.** The engine's free-surface Green functions are {verdict} accuracy of
{tex(d['target'], 0)} at every point of high-precision reference tables: the largest error is
{tex(d['infinite_max'])} in infinite depth and {tex(d['finite_max'])} in finite depth, and the median
errors are smaller by orders of magnitude.

## The problem

A free-surface Green function is the potential of a pulsating source below the free surface that
already satisfies the free-surface, bottom and radiation conditions; every low- and higher-order
result of HydroBEM is built from it, so its accuracy bounds theirs. This page checks it by itself, at
every point of the reference tables (test-plan A2), including the frequency limits:

- in infinite depth, the regular part $F$ and its derivative $F_X$ over $X = KR$ and
  $Y = -K(z + \zeta)$;
- in finite depth, the Green function (real and imaginary parts and their derivatives) over $Kh$ and
  $R/h$.

## The reference

The reference is the integral representations evaluated at 30 digits by
`scripts/gen_green_reference_data.py` and `scripts/gen_green_finite_reference_data.py` (mpmath;
Wehausen & Laitone 1960, John 1950, Newman 1985), stored in `tests/data/green_inf_reference.txt` and
`green_fin_reference.txt`. At 30 digits the reference is exact for this purpose.

The error is relative to $\max(\lvert \text{{value}} \rvert, \text{{the local scale of the singular
parts}})$. The engine's values come from the example `crates/hydrobem/examples/green_error_map.rs`.

## Agreement with the reference

![Green function error maps](../figures/green_error_map.png)

*Figure 1. Left: infinite depth, the error of $F$ and $F_X$ over $X$ and $Y$. Right: finite depth, over
$Kh$ and $R/h$ (ordinary frequencies; the points of the limits are counted in the title).*

Both kernels are {verdict} {e(d['target'])} of theory §8 everywhere (Table 1, Figure 1).

*Table 1. The number of reference points of each kernel and the largest and median error over them.*

| Kernel | points | largest error | median |
|---|---|---|---|
| infinite depth | {d['infinite_points']} | {e(d['infinite_max'])} | {e(d['infinite_median'])} |
| finite depth | {d['finite_points']} | {e(d['finite_max'])} | {e(d['finite_median'])} |
| finite depth, frequency limits | | {e(d['finite_limits_max'])} | |

The summaries of the tables' own error maps are in [development/error-maps/](../../development/error-maps/).

## Reproducing this page

This page, its figure and its numbers are written by `validation/kernels.py` through
`validation/make_report.py`, which builds and runs the example with cargo.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
