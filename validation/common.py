"""Shared helpers of the validation report (docs/validation/README.md and its case pages
docs/validation/cases/): the paths, the problem helpers of the tests (tests/python: meshes.py,
bodies.py, problems.py), the figure style, the store of the numbers the pages quote
(docs/validation/figures/numbers.json) and the helpers that write the pages."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# the engine's warnings of coarse meshes are expected in the convergence studies
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "validation" / "figures"
CASES = ROOT / "docs" / "validation" / "cases"
NUMBERS = FIGURES / "numbers.json"
DATA = ROOT / "tests" / "data"
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests" / "python")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RHO, G = 1000.0, 9.80665
# the series colours, in a fixed order (the reference is always black markers)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REF = {"color": "black", "marker": "o", "ms": 4.5, "mfc": "none", "ls": "none", "label": "reference"}
# values read from a published figure (their read accuracy is stated on the page)
READ = {"color": "black", "marker": "x", "ms": 5.0, "ls": "none", "label": "read from the figure"}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 110, "font.size": 9, "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "legend.fontsize": 8,
})  # fmt: skip


def save(fig, name, tight=True):
    """Writes the figure as docs/validation/figures/<name>.png."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", metadata={"Software": None})
    plt.close(fig)


def e(x, digits=1):
    """A number for the pages in the style 1.2e-3 (0 as 0)."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return "0" if x == 0.0 else "n/a"
    m, p = f"{x:.{digits}e}".split("e")
    return f"{m}e{int(p)}"


def pct(x, digits=1):
    """A fraction as a percentage for the pages."""
    return f"{100.0 * float(x):.{digits}f} %"


def pc(x):
    """A fraction as a percentage of two significant digits for the pages (6.5e-4 as 0.065 %)."""
    return f"{float(f'{100.0 * float(x):.2g}'):g} %"


def tex(x, digits=1):
    """A number in TeX for the prose of the pages, $1.2 \\times 10^{-3}$ (from 0.01 on plain: $0.012$)."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return "$0$" if x == 0.0 else "n/a"
    if abs(x) >= 0.01:
        return f"${float(f'{x:.{digits + 1}g}'):g}$"
    m, p = f"{x:.{digits}e}".split("e")
    return f"${m} \\times 10^{{{int(p)}}}$"


def size(x):
    """A relative difference in words: a percentage from 1e-4 on, a power of ten below."""
    return pc(x) if float(x) >= 1e-4 else tex(x)


def table(header, rows):
    """A Markdown table of the header and the rows (lists of strings)."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


WIDTH = 100  # the line length of the pages


def _units(text):
    """The words of a paragraph, an inline formula $...$, a code span and the text of a link each kept
    as one unit (they stay on one line)."""
    units, current, math_, code, link = [], "", False, False, False
    for word in text.split():
        if word.startswith("%") and not current and units:
            current = units.pop()  # a percent sign stays with its number
        current = f"{current} {word}" if current else word
        math_ ^= word.count("$") % 2 == 1
        code ^= word.count("`") % 2 == 1
        if "[" in word and "](" not in word[word.index("[") :]:
            link = True
        if "](" in word:
            link = False
        if not (math_ or code or link):
            units.append(current)
            current = ""
    return units + ([current] if current else [])


def _fill(text, first, rest):
    """The paragraph filled to WIDTH, its first line after the prefix first, the others after rest."""
    lines, line, empty = [], first, True
    for unit in _units(text):
        if empty:
            line, empty = line + unit, False
        elif len(line) + 1 + len(unit) > WIDTH:
            lines.append(line)
            line = rest + unit
        else:
            line = f"{line} {unit}"
    return "\n".join(lines + [line])


def reflow(text):
    """A page with its prose paragraphs and list items refilled to WIDTH, so that the numbers put in by
    the templates leave no ragged lines and no formula is broken across lines; headings, tables,
    figures and displayed formulas are kept as they are."""
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        if lines[0].startswith(("#", "|", "![", "$$", "```")):
            out.append(block)
        elif lines[0].startswith("- "):
            items, item = [], []
            for line in lines:
                if line.startswith("- ") and item:
                    items.append(item)
                    item = []
                item.append(line[2:] if line.startswith("- ") else line.strip())
            items.append(item)
            out.append("\n".join(_fill(" ".join(i), "- ", "  ") for i in items))
        else:
            out.append(_fill(" ".join(x.strip() for x in lines), "", ""))
    return "\n\n".join(out)


def write_page(name, text):
    """Writes docs/validation/cases/<name>.md (UTF-8, LF)."""
    CASES.mkdir(parents=True, exist_ok=True)
    (CASES / f"{name}.md").write_bytes((text.strip() + "\n").encode("utf-8"))


def wavenumber(omega, depth, g=G):
    """The wavenumber k of omega in the depth, omega^2 = g k tanh kh (deep water when the depth is inf)."""
    nu = np.asarray(omega, dtype=float) ** 2 / g
    if not math.isfinite(depth):
        return nu
    k = np.maximum(nu, np.sqrt(nu / depth))
    for _ in range(60):
        t = np.tanh(k * depth)
        k = k - (k * t - nu) / (t + k * depth * (1.0 - t * t))
    return k


def omega_of(k, depth, g=G):
    """The frequency of the wavenumber k in the depth."""
    k = np.asarray(k, dtype=float)
    return np.sqrt(g * k * (np.tanh(k * depth) if math.isfinite(depth) else 1.0))


def rel_cols(got, ref):
    """Per column the largest |got - ref| over the rows relative to the largest |ref| of the column."""
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    return np.abs(got - ref).max(axis=0) / np.abs(ref).max(axis=0)


def at_read(x, values, points):
    """The values (over x) interpolated linearly to the abscissae of the points read from a figure
    ({x: value}): (abscissae, read values, interpolated values)."""
    xs = np.array(sorted(points))
    return xs, np.array([points[v] for v in xs]), np.interp(xs, x, values)


def read_differences(x, values, read):
    """Per quantity of ``read`` ({name: {x: value}}) the largest |HydroBEM - read| at the points read,
    relative to the largest |read| value of the quantity."""
    out = {}
    for k, pts in read.items():
        _, r, v = at_read(x, values[k], pts)
        out[k] = float(np.abs(v - r).max() / np.abs(r).max())
    return out


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(f"{float(x):.4g}") if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x


def record(group, values):
    """Stores the numbers of a figure group under its name in numbers.json (4 digits)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    old = json.loads(NUMBERS.read_text(encoding="utf-8")) if NUMBERS.exists() else {}
    old[group] = _plain(values)
    NUMBERS.write_bytes((json.dumps(old, indent=1, sort_keys=True) + "\n").encode())


def threads():
    return os.cpu_count() or 1


# the share of the physical memory available that a solve's estimated resident peak may take
MEMORY_SHARE = 0.6


def available_memory():
    """The physical memory available now in bytes (Windows GlobalMemoryStatusEx; MemAvailable of
    /proc/meminfo elsewhere), None where it cannot be read."""
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (n, ctypes.c_uint64) for n in ("total", "avail", "page", "avail_page", "virt", "avail_virt", "ext")
            ]

        s = Status(length=ctypes.sizeof(Status))
        return int(s.avail) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)) else None
    try:
        line = next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable"))
        return int(line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return None


def fit_threads(m):
    """Lowers the model's frequency threads (raising its inner threads to keep the threads in all)
    until the resident peak of its memory estimate takes at most MEMORY_SHARE of the physical memory
    available, so that the studies do not depend on the free memory (the results do not depend on
    the thread counts). Returns the model."""
    avail = available_memory()
    if avail is None:
        return m
    f0 = m.options.frequency_threads or 1
    total = f0 * (m.options.inner_threads or 1)
    f = f0
    while f > 1 and m.memory_estimate().peak > MEMORY_SHARE * avail:
        f = max(1, f // 2)
        m.options.frequency_threads = f
        m.options.inner_threads = max(1, total // f)
    if f != f0:
        print(f"  (frequency threads {f0} -> {f} for the memory available)", flush=True)
    return m


def solve_model(m):
    """The results of the model, its frequency threads fitted to the memory available."""
    return fit_threads(m).solve()


def solve(bodies, omegas, headings=None, depth=math.inf, environment=None, **kw):
    """problems.solve with the frequency threads fitted to the memory available."""
    import problems

    env = {"depth": depth, "density": problems.RHO, "gravity": problems.G, **(environment or {})}
    return solve_model(problems.model(bodies, omegas, headings, environment=env, **kw))


def peak(a):
    return float(np.abs(a).max())


def rel(a, b):
    """The largest |a - b| relative to the largest |b|."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def observed_order(v, h):
    """The order p and the Richardson extrapolation of three values v on the sizes h (h0 > h1 > h2),
    per entry: v_i = v* + c h_i^p."""
    v0, v1, v2 = (np.asarray(x, dtype=float) for x in v)
    d1, d2 = v1 - v0, v2 - v1
    ps = np.linspace(0.3, 6.0, 5701)
    ratio = (h[0] ** ps - h[1] ** ps) / (h[1] ** ps - h[2] ** ps)  # increases with p
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d2 != 0, d1 / d2, np.nan)
    p = np.interp(np.clip(q, ratio.min(), ratio.max()), ratio, ps)
    extrap = v2 + d2 / ((h[1] / h[2]) ** p - 1.0)
    return p, extrap


def run_isolated(module, function, env=None, **kwargs):
    """Runs validation/<module>.py's function(**kwargs) in a process of its own with the
    environment variables env added (the validation hooks HB_VALIDATE_* of the engine are read once
    per process) and returns its JSON result."""
    code = (f"import json, math, sys; sys.path.insert(0, {str(ROOT / 'validation')!r}); import {module} as m; "
            f"print('@@' + json.dumps(m.{function}(**{kwargs!r})))")  # fmt: skip
    e = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
    out = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"{module}.{function} {env}: {out.stderr[-2000:]}")
    line = next(s for s in out.stdout.splitlines() if s.startswith("@@"))
    return json.loads(line[2:])


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t
        print(f"  {self.name}: {self.seconds:.0f} s", flush=True)
