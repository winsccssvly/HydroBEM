"""The item names in the messages of the rules exist in the schema.

A rule names the items it refuses (`first_order.motion = true requires ...`,
`control_surface_waterline_term = 'body' needs second_order.drift_control_surface`). When an item
is renamed, a message that still names the old item compiles and passes every test that does not
trip it. This test reads the string literals of the engine's sources (the rule layers of
docs/development/architecture.md, "Where a rule goes") and checks the names they give:

- `section.item`, where `section` is a single object of the model (`first_order`,
  `second_order`, `options`, `environment`): `item` is an item of its type (followed by a
  placeholder of `format!` or by `...`, as in `second_order.free_surface_{name}`: the start of
  one);
- `name = value` with a value `true`, `false`, a quoted text or a list, or a `name` with an
  underscore: `name` is an item of some type of the schema (`x = 0`, `omega = inf` are not names
  of items).
"""

import re
from pathlib import Path

import pytest

from hydrobem.objects import schema

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "crates" / "hydrobem" / "src"

# the files of the rule layers (a directory: every file in it); a moved rule keeps its file's
# directory or is added here
RULE_FILES = [
    "document.rs",
    "document_rules.rs",
    "document_build.rs",
    "document_enabled.rs",
    "problem.rs",
    "rules",
    "field_points.rs",
    "plate_rules.rs",
    "tanks.rs",
    "route.rs",
    "quality.rs",
    "memory.rs",
    "model_queries.rs",
    "model_solve.rs",
    "potential_qtf/zones.rs",
]

LITERAL = re.compile(r'(?<![A-Za-z0-9_\'])"((?:[^"\\]|\\.)*)"', re.S)
DOTTED = re.compile(r"(?<![\w.])([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b(\{|\.\.\.)?")
ASSIGNED = re.compile(r"(?<![\w.])([a-z_][a-z0-9_]*) = (true|false|'|\[|\\\")?")

pytestmark = pytest.mark.skipif(not SRC.is_dir(), reason="the engine's sources (a checkout)")


def rule_sources() -> list[Path]:
    out = []
    for name in RULE_FILES:
        p = SRC / name
        if p.is_dir():
            out.extend(sorted(p.rglob("*.rs")))
        elif p.is_file():
            out.append(p)
    return out


def literals(path: Path) -> list[tuple[int, str]]:
    """(line, text) of every string literal of a Rust file outside its comments; a literal
    continued with a backslash at the end of a line is joined."""
    text = path.read_text(encoding="utf-8")
    # comments to blanks of the same length, so that the line numbers stay
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    out = []
    for m in LITERAL.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        out.append((line, re.sub(r"\\\n\s*", "", m.group(1))))
    return out


def model_sections(s: dict) -> dict[str, str]:
    """The single objects of the model (`first_order`, ...) and their types."""
    return {name: lst["types"][0] for name, lst in s["Model"]["lists"].items() if lst["single"]}


def test_the_rule_files_exist():
    # a file of the list that is gone would leave its messages unchecked
    found = rule_sources()
    assert SRC / "rules" / "mod.rs" in found and SRC / "document_build.rs" in found
    assert [n for n in RULE_FILES if not (SRC / n).exists()] == []


def test_every_item_named_in_a_rule_message_exists():
    s = schema()
    sections = model_sections(s)
    assert {"first_order", "second_order", "options", "environment"} <= set(sections)
    items = {t: set(entry["items"]) for t, entry in s.items()}
    every = set().union(*items.values())
    wrong, checked = [], 0
    for path in rule_sources():
        where = path.relative_to(SRC).as_posix()
        for line, text in literals(path):
            for section, item, more in DOTTED.findall(text):
                if section in sections:
                    checked += 1
                    known = items[sections[section]]
                    if not (any(k.startswith(item) for k in known) if more else item in known):
                        wrong.append(f"{where}:{line}: {section}.{item}")
            for name, value in ASSIGNED.findall(text):
                if value or "_" in name:
                    checked += 1
                    if name not in every:
                        wrong.append(f"{where}:{line}: {name} = ...")
    assert wrong == []
    # the scan finds the names (a change of the patterns or the files that finds none fails)
    assert checked > 50
