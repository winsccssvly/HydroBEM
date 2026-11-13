"""Rewrites the citations of the documentation as it was before its restructuring by reader and
topic (2026-11-12) into the present form, and checks the citations and links of the documents.

    python scripts/rewrite_doc_refs.py [--check] [--table] [file ...]

Before, the documentation had one file per kind (`docs/theory/physics.md`, `numerics.md`,
`second-order.md`, `docs/user-guide.md`, `docs/reference/inputs.md`,
`results-and-files.md`, `docs/generated/`) and the code cited them as `physics §12`,
`numerics §3.2a`, `second-order §4a`, `inputs §8.1`, `results-and-files §1.1`. Now it is
divided by reader (`docs/guide/`, `docs/theory/`, `docs/reference/`, ...), and a citation names
the folder and the file (docs/development/rules.md):

- `theory §11.2`: section 2 of chapter 11 of the theory (`docs/theory/11-*.md`); within the
  theory the bare `§11.2`;
- `guide/<file> §N`, `reference/<file> §N`: section N of `docs/guide/<file>.md` or
  `docs/reference/<file>.md` (the file alone without §); within the same file the bare `§N`.

Without arguments every tracked text file is rewritten (the code, the documents, the scripts,
the tests, the validation report, the changelog), except the dated records of the history of
the development repository (not published), whose old citations are looked up in the table
`--table` writes. The rewrite is idempotent: what it writes is never an old form.

--check rewrites nothing and fails if a file cites an old form or a relative link of a published
Markdown file (docs/, README.md, ...) names a missing file or heading.
--table prints the table of the old sections and where they went (Markdown).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/rewrite_doc_refs.py"
# the project instructions are the owner's to change
OWNER = ("CLAUDE" + ".md",)
SKIP_DIRS = ("licenses/", "tests/data/")
TEXT = (".md", ".py", ".rs", ".h", ".c", ".bat", ".sh", ".toml", ".yml", ".cff", ".txt")

OLD_FILES = {
    "physics": "docs/theory/physics.md",
    "numerics": "docs/theory/numerics.md",
    "second-order": "docs/theory/second-order.md",
    "inputs": "docs/reference/inputs.md",
    "results-and-files": "docs/reference/results-and-files.md",
    "user-guide": "docs/user-guide.md",
}
# where a citation of a whole old file goes
DEFAULT = {
    "physics": "theory:",
    "numerics": "theory:",
    "second-order": "theory:12",
    "inputs": "reference/inputs-model:",
    "results-and-files": "reference/results:",
    "user-guide": "guide/getting-started:",
}
MOVED = {
    "docs/generated/green_inf_error_map.md": "docs/development/error-maps/green_inf_error_map.md",
    "docs/generated/green_fin_error_map.md": "docs/development/error-maps/green_fin_error_map.md",
    "docs/generated/panel_rule_error_map.md": "docs/development/error-maps/panel_rule_error_map.md",
}
MOVED_DIRS = {"docs/generated": "docs/development/error-maps"}

# (old document, old section, old heading, new place): a new place is `theory:C.S` (chapter C,
# section S), `guide/<file>:N` or `reference/<file>:N` (N empty: the file)
SECTIONS = [
    ("physics", "1", "1. Coordinates and conventions", "theory:1.1"),
    ("physics", "2", "2. Field equations and boundary conditions", "theory:1.2"),
    ("physics", "3", "3. Radiation, excitation and coefficients", "theory:2.1"),
    ("physics", "4", "4. Frequency limits ($\\omega = 0$ and $\\omega = \\infty$)", "theory:1.3"),
    ("physics", "5", "5. Physical damping lids", "theory:9.1"),
    ("physics", "6", "6. Prescribed-pressure surfaces (air cushions, chambers)", "theory:9.3"),
    ("physics", "7", "7. Thin (dipole) sheets", "theory:9.5"),
    ("physics", "8", "8. Rigid-body motions", "theory:3.1"),
    ("physics", "9", "9. Hydrostatics", "theory:3.2"),
    ("physics", "10", "10. Free-surface elevation, field pressure and velocity", "theory:4.1"),
    ("physics", "11", "11. Irregular frequencies", "theory:5.1"),
    ("physics", "12", "12. Mean drift force and moment (momentum conservation)", "theory:11.1"),
    ("physics", "13", "13. Mean drift force and moment (pressure integration)", "theory:11.2"),
    ("physics", "14", "14. Generalized modes", "theory:10.1"),
    ("physics", "15", "15. Internal tanks", "theory:10.2"),
    ("physics", "16", "16. Mean drift force and moment (control surface)", "theory:11.3"),
    ("physics", "17", "17. Wavemakers in the walls", "theory:10.3"),
    ("physics", "18", "18. Rigid plates in the free surface", "theory:9.7"),
    ("numerics", "1", "1. Low-order method", "theory:6"),
    ("numerics", "1.1", "1.1 Rankine integrals", "theory:6.1"),
    ("numerics", "1.2", "1.2 The logarithmic singularity of the wave kernel", "theory:6.2"),
    ("numerics", "1.3", "1.3 Symmetry sectors and walls", "theory:6.3"),
    ("numerics", "2", "2. Higher-order method", "theory:7"),
    ("numerics", "2.1", "2.1 Static (Rankine) integrals, by tiers", "theory:7.1"),
    ("numerics", "2.2", "2.2 Wave part and right-hand sides", "theory:7.2"),
    ("numerics", "2.3", "2.3 Discretization from a panel size", "theory:7.3"),
    ("numerics", "3", "3. Free-surface Green function", "theory:8"),
    ("numerics", "3.1", "3.1 Infinite depth", "theory:8.1"),
    ("numerics", "3.2", "3.2 Finite depth", "theory:8.2"),
    ("numerics", "3.2a", "3.2a Four points at a time", "theory:8.3"),
    ("numerics", "3.3", "3.3 Frequency limits", "theory:8.4"),
    ("numerics", "3.4", "3.4 Walls and channel", "theory:8.5"),
    ("numerics", "4", "4. Interior surfaces and lids", "theory:5.2"),
    ("numerics", "5", "5. Thin sheets", "theory:9.6"),
    ("numerics", "6", "6. Pressure surfaces", "theory:9.4"),
    ("numerics", "7", "7. Linear algebra", "theory:13.1"),
    ("numerics", "8", "8. Hydrostatics and motion matrices", "theory:3.3"),
    ("numerics", "9", "9. Accuracy targets of complete results", "theory:13.2"),
    ("numerics", "10", "10. Velocities and pressure at points", "theory:4.2"),
    ("numerics", "11", "11. Mean drift", "theory:11.4"),
    ("numerics", "12", "12. Geometry preprocessing and tolerances", "theory:14"),
    ("numerics", "13", "13. Threads, determinism and memory", "theory:13.3"),
    ("second-order", "1", "1. Pairs and coefficients", "theory:12.2"),
    ("second-order", "2", "2. The quadratic part", "theory:12.3"),
    ("second-order", "3", "3. The second-order potential", "theory:12.4"),
    ("second-order", "4", "4. The indirect method", "theory:12.5"),
    ("second-order", "4a", "4a. The direct method", "theory:12.6"),
    ("second-order", "5", "5. The body condition of a floating body", "theory:12.7"),
    ("second-order", "6", "6. The free surface inside the radius R", "theory:12.8"),
    ("second-order", "7", "7. The far field beyond R", "theory:12.9"),
    ("second-order", "8", "8. The sub-problems", "theory:12.10"),
    ("second-order", "9", "9. Accuracy and cost", "theory:12.12"),
    ("second-order", "10", "10. Newman's approximation", "theory:12.13"),
    ("second-order", "11", "11. One wall", "theory:12.14"),
    ("second-order", "12", "12. Generalized modes", "theory:12.15"),
    ("inputs", "1", "1. Geometry", "reference/inputs-meshes:"),
    ("inputs", "1.1", "1.1 Low-order mesh (kind `panels`)", "reference/inputs-meshes:1"),
    ("inputs", "1.2", "1.2 Higher-order mesh (kinds `flat_patches`, `spline_patches`, `analytic_patches`)",
     "reference/inputs-meshes:2"),
    ("inputs", "1.3", "1.3 The items `mesh` and `ranges`", "reference/inputs-meshes:3"),
    ("inputs", "1.4", "1.4 The queries of a mesh", "reference/inputs-meshes:4"),
    ("inputs", "1.5", "1.5 The geometry file (.hbg)", "reference/inputs-meshes:5"),
    ("inputs", "2", "2. Body", "reference/inputs-bodies:1"),
    ("inputs", "2.1", "2.1 Items", "reference/inputs-bodies:1.1"),
    ("inputs", "2.2", "2.2 Trim", "reference/inputs-bodies:1.2"),
    ("inputs", "2.3", "2.3 Couplings between bodies", "reference/inputs-bodies:1.3"),
    ("inputs", "3", "3. The surfaces of a body", "reference/inputs-bodies:2"),
    ("inputs", "3.1", "3.1 Dipoles and dipole dampers", "reference/inputs-bodies:2.1"),
    ("inputs", "3.2", "3.2 Tanks", "reference/inputs-bodies:2.2"),
    ("inputs", "3.3", "3.3 Prescribed-pressure surfaces", "reference/inputs-bodies:2.3"),
    ("inputs", "3.3a", "3.3a Free-surface plates", "reference/inputs-bodies:2.4"),
    ("inputs", "3.4", "3.4 Generalized modes", "reference/inputs-bodies:2.5"),
    ("inputs", "3.5", "3.5 Interior surface (irregular frequencies)", "reference/inputs-bodies:2.6"),
    ("inputs", "3.6", "3.6 Control surface", "reference/inputs-bodies:2.7"),
    ("inputs", "4", "4. Lids, fixed dampers, field points and the free surface of the potential QTF",
     "reference/inputs-model-objects:"),
    ("inputs", "4.1", "4.1 Damping lids (`m.lids`, `DampingLid`)", "reference/inputs-model-objects:1"),
    ("inputs", "4.2", "4.2 Fixed dampers (`m.fixed_dampers`, `FixedDamper`)", "reference/inputs-model-objects:2"),
    ("inputs", "4.3", "4.3 Field points (`first_order.field_points`, a tank's `field_points`)",
     "reference/inputs-model-objects:3"),
    ("inputs", "4.4", "4.4 Free surface of the potential QTF (items of `second_order`)", "reference/inputs-orders:3"),
    ("inputs", "5", "5. Environment: the water and the waves", "reference/inputs-model:1"),
    ("inputs", "6", "6. Waves", "reference/inputs-model:2"),
    ("inputs", "7", "7. Options, first and second order", "reference/inputs-orders:"),
    ("inputs", "8", "8. Model rules (validation)", "reference/inputs-rules:1"),
    ("inputs", "8.1", "8.1 Supported combinations", "reference/inputs-rules:2"),
    ("inputs", "9", "9. The model file", "reference/inputs-model:4"),
    ("inputs", "10", "10. Queries", "reference/queries:"),
    ("inputs", "11", "11. Suppressed objects (`enabled`)", "reference/inputs-model:5"),
    ("results-and-files", "1", "1. `Results` (Python)", "reference/results:"),
    ("results-and-files", "1.1", "1.1 Results by owner and by name", "reference/results:1"),
    ("results-and-files", "1.2", "1.2 The results file of the engine (.hbr)", "reference/results:2"),
    ("results-and-files", "2", "2. Normalization (files only)", "reference/file-formats:1"),
    ("results-and-files", "3", "3. Numeric files", "reference/file-formats:2"),
    ("results-and-files", "4", "4. The standalone `.out`", "reference/file-formats:3"),
    ("results-and-files", "5", "5. Writing rules", "reference/file-formats:4"),
    # the user guide had no section numbers: its sections by heading
    ("user-guide", "", "Installation", "guide/getting-started:1"),
    ("user-guide", "", "Troubleshooting", "guide/getting-started:1.1"),
    ("user-guide", "", "Concepts", "guide/getting-started:2"),
    ("user-guide", "", "A first problem", "guide/getting-started:3"),
    ("user-guide", "", "Geometry", "guide/meshes-and-bodies:1"),
    ("user-guide", "", "Bodies", "guide/meshes-and-bodies:2"),
    ("user-guide", "", "Surfaces of a body", "guide/special-surfaces:1"),
    ("user-guide", "", "Trim", "guide/meshes-and-bodies:4"),
    ("user-guide", "", "Suppressing an object", "guide/meshes-and-bodies:5"),
    ("user-guide", "", "Lids, fixed dampers and field points", "guide/special-surfaces:2"),
    ("user-guide", "", "Environment, options, first and second order", "guide/environment-and-options:1"),
    ("user-guide", "", "Supported combinations", "guide/environment-and-options:2"),
    ("user-guide", "", "Mean drift", "guide/second-order-loads:1"),
    ("user-guide", "", "Results", "guide/results:1"),
    ("user-guide", "", "Model and results files", "guide/getting-started:5"),
    ("user-guide", "", "The model in the engine", "guide/getting-started:4"),
    ("user-guide", "", "File formats", "guide/results:2"),
    ("user-guide", "", "Performance", "guide/performance:1"),
]
# old sections split between two chapters: the place by the words of the citing line
CONTEXT = {
    ("physics", "13"): [(r"(?i)\bqtfs?\b|quadratic|second-order potential", "theory:12.1")],
    ("numerics", "11"): [(r"(?i)\bqtfs?\b|quadratic|second-order potential", "theory:12.11")],
    ("numerics", "4"): [(r"(?i)\blids?\b", "theory:9.2"), (r"(?i)\bplates?\b", "theory:9.8")],
    ("inputs", "7"): [(r"(?i)\boptions\b", "reference/inputs-model:3")],
}
LOC = {(d, s): loc for d, s, _, loc in SECTIONS if s}

ITEM = r"\d+[a-z]?(?:\.\d+[a-z]?)?"
SEP = r"(?:, ?§ ?| and §| or §|[–-]§?)"
CHAIN = rf"§ ?(?P<chain>{ITEM}(?:{SEP}{ITEM})*)"
DOCS = r"(?P<doc>[Pp]hysics|numerics|second-order|inputs|results-and-files)"
# `physics §12`, `inputs.md §8.1`
CITATION = re.compile(rf"(?<![\w/.-]){DOCS}(?P<md>\.md)?(?P<sp> ?){CHAIN}")
WRAPPED = re.compile(rf"(?<![\w/.-]){DOCS}(?P<md>\.md)?(?P<nl>\n[ \t]*(?://[/!]?|#|\*|rem)?[ \t]*){CHAIN}")
# `[second-order.md](second-order.md) §4–§9`
LINK_CHAIN = re.compile(rf"\[(?P<text>[^\]\n]*)\]\((?P<target>[^)\s]+)\)(?P<sp> ?){CHAIN}")
LINK = re.compile(r"(?<!!)\[(?P<text>[^\]\n]*)\]\((?P<target>[^)\s]+)\)")
OLD_PATH = re.compile(r"(?<![\w/.-])(?<!\]\()(?P<q0>`?)(?P<path>(?:\.\./)*(?:docs/)?(?:(?:theory|reference)/)?"
                      r"(?P<name>physics|numerics|second-order|inputs|results-and-files|user-guide)\.md)"
                      r"(?P<anchor>#[\w-]+)?(?P<q>`?)(?P<tail>(?P<sp> ?)" + CHAIN + ")?")
OLD_GENERATED = re.compile(r"(?<![\w-])(?P<pre>(?:\.\./)*(?:docs/)?)generated/(?P<file>\w+\.md)?")
# what --check refuses outside the history
OLD_FORMS = [CITATION, WRAPPED, re.compile(r"(?<![\w-])user-guide\.md"), re.compile(r"docs/generated\b"),
             re.compile(r"(?<![\w-])(?:physics|numerics|second-order|results-and-files)\.md"),
             re.compile(r"(?<![\w-])reference/inputs\.md"), re.compile(r"\binputs\.md\b")]


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading."""
    text = re.sub(r"[`*]", "", heading.strip()).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(text: str) -> list[tuple[str, str]]:
    """(heading text, anchor) of every heading of a Markdown text, outside fenced blocks."""
    out, seen, fenced = [], {}, False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        m = None if fenced else re.match(r"#{1,6} (.+)", line)
        if m:
            a = slug(m.group(1))
            n = seen.get(a, 0)
            seen[a] = n + 1
            out.append((m.group(1).strip(), a if n == 0 else f"{a}-{n}"))
    return out


OLD_ANCHORS = {(d, slug(h)): loc for d, _, h, loc in SECTIONS}


def loc_file(loc: str) -> str:
    """The repository path of a new place."""
    where = loc.split(":")[0]
    if where == "theory":
        sec = loc.split(":")[1]
        if not sec:
            return "docs/theory/README.md"
        chapter = int(sec.split(".")[0])
        found = sorted((ROOT / "docs" / "theory").glob(f"{chapter:02d}-*.md"))
        return found[0].relative_to(ROOT).as_posix() if found else f"docs/theory/{chapter:02d}.md"
    return f"docs/{where}.md"


def loc_sec(loc: str) -> str:
    return loc.split(":")[1]


@lru_cache(maxsize=None)
def file_anchors(path: str) -> tuple[tuple[str, str], ...]:
    p = ROOT / path
    return tuple(anchors(p.read_text(encoding="utf-8"))) if p.exists() else ()


def loc_anchor(loc: str) -> str:
    """The anchor of the heading of a new place ('' for a file, or if it has none)."""
    sec = loc_sec(loc)
    if not sec or (loc.startswith("theory:") and "." not in sec):
        return ""
    for heading, a in file_anchors(loc_file(loc)):
        if re.match(rf"{re.escape(sec)}\.? ", heading):
            return a
    return ""


def cite(loc: str, here: str) -> str:
    """The citation of a new place in the file `here`."""
    where, sec = loc.split(":")
    if where == "theory":
        if here.startswith("docs/theory/") and sec:
            return f"§{sec}"
        return f"theory §{sec}" if sec else "theory"
    if loc_file(loc) == here and sec:
        return f"§{sec}"
    return f"{where} §{sec}" if sec else where


def place(doc: str, sec: str, line: str) -> str | None:
    for pattern, loc in CONTEXT.get((doc, sec), []):
        if re.search(pattern, line):
            return loc
    return LOC.get((doc, sec))


def chain_text(doc: str, chain: str, here: str, line: str, unknown: list) -> tuple[str, str] | None:
    """The new citation of an old chain of sections (`§6, §8`, `§4–§9`), and its first place."""
    parts = re.split(rf"({SEP})", chain)
    out, prev, first = [], None, None
    for k in range(0, len(parts), 2):
        loc = place(doc, parts[k], line)
        if loc is None:
            unknown.append(f"{doc} §{parts[k]}")
            return None
        first = first or loc
        text = cite(loc, here)
        # one document: the theory, or a file of the guide or the reference
        doc_of = "theory" if loc.startswith("theory:") else loc_file(loc)
        if prev is not None:
            sep = parts[k - 1]
            same = prev == doc_of
            if same and loc_sec(loc):
                text = f"§{loc_sec(loc)}"
            sep = ", " if "," in sep else " and " if "and" in sep else " or " if "or" in sep else (
                sep[0] if same else " to ")
            out.append(sep)
        out.append(text)
        prev = doc_of
    return "".join(out), first


def relative(target: str, here: str) -> str:
    return os.path.relpath(ROOT / target, (ROOT / here).parent).replace(os.sep, "/")


def new_target(loc: str, here: str) -> str:
    path, a = loc_file(loc), loc_anchor(loc)
    if path == here:
        return f"#{a}" if a else ""
    return relative(path, here) + (f"#{a}" if a else "")


def old_doc_of(target: str, here: str) -> tuple[str | None, str]:
    """The old document a link target names (None if not one) and its anchor."""
    path, _, anchor = target.partition("#")
    if not path:
        return None, anchor
    full = os.path.normpath(os.path.join((ROOT / here).parent, path))
    rel = Path(full).relative_to(ROOT).as_posix() if Path(full).is_relative_to(ROOT) else ""
    for doc, old in OLD_FILES.items():
        if rel == old:
            return doc, anchor
    return None, anchor


def text_loc(text: str, here: str) -> str | None:
    """The new place a (rewritten) link text cites."""
    m = re.search(r"\b(guide|reference)/([\w-]+)(?: §(\d+(?:\.\d+)?))?", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}:{m.group(3) or ''}"
    m = re.search(r"theory §(\d+(?:\.\d+)?)", text)
    if m is None and here.startswith("docs/theory/"):
        m = re.search(r"§(\d+(?:\.\d+)?)", text)
    if m:
        return f"theory:{m.group(1)}"
    return None


def rewrite(text: str, here: str, unknown: list) -> str:
    """The text of the file `here` with every old citation and link in the present form."""
    is_md = here.endswith(".md")

    # a citation broken across two lines (`physics` at the end of one, `§16` at the start of the
    # next, after the comment marker of the code)
    def wrapped(m):
        r = chain_text(m.group("doc").lower(), m.group("chain"), here, m.group(0), unknown)
        if r is None:
            return m.group(0)
        head, _, rest = r[0].partition(" §")
        return head + m.group("nl") + ("§" + rest if rest else "")

    text = WRAPPED.sub(wrapped, text)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if is_md:
            # a link followed by sections: [second-order.md](second-order.md) §4–§9
            def link_chain(m, line=line):
                doc, _ = old_doc_of(m.group("target"), here)
                if doc is None:
                    return m.group(0)
                r = chain_text(doc, m.group("chain"), here, line, unknown)
                if r is None:
                    return m.group(0)
                return f"[{r[0]}]({new_target(r[1], here) or '#'})"

            line = LINK_CHAIN.sub(link_chain, line)

        def citation(m, line=line):
            r = chain_text(m.group("doc").lower(), m.group("chain"), here, line, unknown)
            if r is None:
                return m.group(0)
            text = r[0]
            if m.group("doc")[0] == "P" and text[0].islower():
                text = text[0].upper() + text[1:]
            return text

        line = CITATION.sub(citation, line)
        # plain paths (not link targets): `docs/reference/inputs.md` §8, docs/user-guide.md
        def path(m, line=line):
            name = m.group("name")
            if m.group("tail"):
                r = chain_text(name, m.group("chain"), here, line, unknown)
                if r is None:
                    return m.group(0)
                return m.group("q0") + r[0] + m.group("q") if m.group("q0") != m.group("q") else r[0]
            loc = OLD_ANCHORS.get((name, m.group("anchor")[1:])) if m.group("anchor") else None
            new = loc_file(loc or DEFAULT[name])
            old = m.group("path")
            if old.startswith("docs/"):
                shown = new
            elif "../" in old:
                shown = relative(new, here)
            elif "/" in old:
                shown = new[len("docs/"):]
            else:
                shown = new.rsplit("/", 1)[1]
            return m.group("q0") + shown + m.group("q")

        line = OLD_PATH.sub(path, line)
        if is_md:
            def link(m):
                target = m.group("target")
                if "://" in target or target.startswith("mailto:"):
                    return m.group(0)
                doc, anchor = old_doc_of(target, here)
                text = m.group("text")
                if doc is None:
                    path = target.partition("#")[0]
                    full = os.path.normpath(os.path.join((ROOT / here).parent, path)) if path else ""
                    rel = Path(full).relative_to(ROOT).as_posix() if full and Path(full).is_relative_to(ROOT) else ""
                    for old, new in {**MOVED, **MOVED_DIRS}.items():
                        if rel.rstrip("/") == old.rstrip("/"):
                            t = relative(new, here) + ("/" if path.endswith("/") else "")
                            if re.fullmatch(r"[\w./-]*generated/?[\w.]*", text):
                                text = t
                            return f"[{text}]({t})"
                    return m.group(0)
                loc = OLD_ANCHORS.get((doc, anchor)) if anchor else None
                if loc is None and anchor:
                    loc = next((v for (d, a), v in OLD_ANCHORS.items() if d == doc and a.startswith(anchor)), None)
                loc = loc or text_loc(text, here) or DEFAULT[doc]
                t = new_target(loc, here) or "#"
                if re.fullmatch(r"[\w./-]+\.md", text):  # the link text is the file name
                    text = loc_file(loc)[len("docs/"):]
                return f"[{text}]({t})"

            line = LINK.sub(link, line)


        def generated(m):
            return m.group(0).replace("generated/", "development/error-maps/") if (m.group("pre") or m.group(
                "file")) else m.group(0)

        line = OLD_GENERATED.sub(generated, line)
        lines[i] = line
    return "\n".join(lines)


def tracked() -> list[str]:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in files.splitlines()
            if f.endswith(TEXT) and not f.startswith(SKIP_DIRS) and f != SELF and f not in OWNER
            and (ROOT / f).exists()]


PUBLISHED_MD = ("docs/", "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "examples/README.md")


def check_links(path: str, text: str) -> list[str]:
    """The relative links of a Markdown file that name a missing file or heading."""
    errors, fenced = [], False
    own = {a for _, a in anchors(text)}
    for no, line in enumerate(text.split("\n"), 1):
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"(`+).+?\1", "", line)
        for m in LINK.finditer(line):
            target = m.group("target")
            if re.match(r"[a-z]+:", target):
                continue
            file, _, anchor = target.partition("#")
            if not file:
                if anchor not in own:
                    errors.append(f"{path}:{no}: no heading #{anchor}")
                continue
            p = (ROOT / path).parent / file
            if not p.exists():
                errors.append(f"{path}:{no}: no file {file}")
            elif anchor and p.suffix == ".md":
                if anchor not in {a for _, a in anchors(p.read_text(encoding="utf-8"))}:
                    errors.append(f"{path}:{no}: no heading #{anchor} in {file}")
    return errors


def table() -> str:
    rows = ["| Old citation | Old heading | New citation | New file |", "|---|---|---|---|"]
    for doc, sec, heading, loc in SECTIONS:
        old = f"{doc} §{sec}" if sec else f"{doc} ({heading})"
        new = cite(loc, "")
        rows.append(f"| `{old}` | {heading.replace('|', '/')} | `{new}` | `{loc_file(loc)}` |")
    for (doc, sec), alts in CONTEXT.items():
        for pattern, loc in alts:
            words = pattern.replace("|", " or ")
            rows.append(f"| `{doc} §{sec}` (a citing line with {words}) | | `{cite(loc, '')}` | `{loc_file(loc)}` |")
    for old, new in {**MOVED, **MOVED_DIRS}.items():
        rows.append(f"| `{old}` | | `{new}` | |")
    return "\n".join(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if "--table" in args:
        print(table())
        return
    check = "--check" in args
    files = [a for a in args if not a.startswith("--")] or tracked()
    problems, unknown, changed = [], [], 0
    for f in files:
        f = Path(f).resolve().relative_to(ROOT).as_posix() if Path(f).is_absolute() else f
        p = ROOT / f
        text = p.read_bytes().decode("utf-8")
        if check:
            for pattern in OLD_FORMS:
                for m in pattern.finditer(text):
                    problems.append(f"{f}:{text.count(chr(10), 0, m.start()) + 1}: old citation {m.group(0)!r}")
            if f.endswith(".md") and f.startswith(PUBLISHED_MD):
                problems += check_links(f, text)
            continue
        new = rewrite(text, f, unknown)
        if new != text:
            p.write_bytes(new.encode("utf-8"))
            changed += 1
            print(f"rewritten: {f}")
    for u in sorted(set(unknown)):
        print(f"unknown section: {u}")
    if check:
        for e in problems:
            print(e)
        if problems:
            print(f"rewrite_doc_refs: {len(problems)} findings", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"{changed} files rewritten")


if __name__ == "__main__":
    main()
