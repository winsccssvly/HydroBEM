"""Writes the regression baselines tests/data/regression/<problem>.npz.

    python scripts/gen_regression_baselines.py [problem ...]

The problems are those of tests/python/regression_problems.py; each file
holds every numeric output of HydroBEM for its problem (with the threads of
the problem: the results do not depend on them), keyed as in
regression_problems.outputs. Run it only after an intended change
of results, and give the reason in the commit.
"""

import datetime as dt
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests" / "python")]

import hydrobem as hb  # noqa: E402

from regression_problems import PROBLEMS, solve  # noqa: E402

OUT = ROOT / "tests" / "data" / "regression"


def history() -> list[str]:
    """The entries of the earlier writes kept in README.md (a partial write adds to them)."""
    readme = OUT / "README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8").splitlines()
    if "## Writes" not in text:
        return []
    out = []
    for x in text[text.index("## Writes") + 1 :]:
        if x.startswith("- "):  # one line per write
            out.append(x)
        elif x.startswith("#"):
            break
    return out


def notes() -> list[str]:
    """The sections of README.md after the writes (e.g. the changes without a write), kept."""
    readme = OUT / "README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8").splitlines()
    k = next((i for i, x in enumerate(text) if x.startswith("## ") and x != "## Writes"), None)
    return [] if k is None else ["", *text[k:]]


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    earlier = history()
    kept = notes()
    for name in names or PROBLEMS:
        out = solve(name)
        np.savez_compressed(OUT / f"{name}.npz", **out)
        print(f"{name}: {len(out)} arrays (HydroBEM {hb.__version__})")
    lines = [
        "# Regression baselines",
        "",
        "HydroBEM's own results of the problems of `tests/python/regression_problems.py`, one",
        "`.npz` per problem, written by `scripts/gen_regression_baselines.py` and checked by",
        "`tests/python/test_regression.py`. Do not edit; regenerate only after an intended",
        "change of results and give the reason in the commit.",
        "",
        "## Writes",
        "",
        *earlier,
        # each write with the machine and the linear-algebra library it ran on
        f"- {dt.date.today().isoformat()}, HydroBEM {hb.__version__}, {platform.system()}, NumPy {np.__version__}, "
        f"{platform.processor() or platform.machine()}, {hb.build_info().linear_algebra}: "
        f"{'all problems' if not names else ', '.join(names)}.",
        *kept,
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1:])
