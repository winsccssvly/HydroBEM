"""Extracts the A4 reference records (test-plan A4) from the published
tables of A. Hulme (1982), The wave forces acting on a floating hemisphere
undergoing forced periodic oscillations, J. Fluid Mech. 121, 443-463,
https://doi.org/10.1017/S0022112082001980: table 1 (heave, A(0), B(0)) and
table 2 (surge, A(1), B(1)) at every Ka > 0 listed in both tables, 0.1 to 10
(tests/data/hulme1982_full.txt; A4 takes its rows Ka = 0.5, 1.0, 2.0, the wider range is that
of A4 with irregular-frequency removal), and the limits of the added masses at Ka = 0 and
Ka = inf (tests/data/hulme1982_limits.txt): heave at Ka = 0 the constant L of eq. (3.18), at
Ka = inf the row inf of table 1; surge at Ka = 0 the row 0 of table 2 (eq. 4.21), at Ka = inf the
constant C1 of eq. (5.15) (the row inf of table 2 is C1 to four digits).

The input is a local Markdown conversion of the paper (every row and both constants checked
against the PDF; the paper itself is not part of the repository); the numbers are parsed, not
typed.

Usage: python scripts/extract_hulme.py <Markdown conversion of the paper>
Writes tests/data/hulme1982_full.txt and tests/data/hulme1982_limits.txt (ASCII, LF line ends).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FULL = ROOT / "tests" / "data" / "hulme1982_full.txt"
OUT_LIMITS = ROOT / "tests" / "data" / "hulme1982_limits.txt"


def table_rows(lines, caption):
    """Rows 'Ka | A | B' of the Markdown table that precedes or follows the caption ('inf' for
    the row of Ka = infinity)."""
    idx = next(i for i, line in enumerate(lines) if caption in line)
    rows = {}
    # the tables sit within 60 lines before (table 1) or after (table 2) their captions
    for line in lines[max(0, idx - 60) : idx + 60]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] == r"$\infty$":
            cells[0] = "inf"
        if len(cells) == 3 and re.fullmatch(r"\d+(\.\d+)?|inf", cells[0]):
            try:
                rows[cells[0]] = (float(cells[1]), float(cells[2]))
            except ValueError:
                continue
    return rows


def constant(text, pattern):
    """The number of the first match of the pattern (its group 1)."""
    return float(re.search(pattern, text).group(1))


def main():
    src = Path(sys.argv[1])
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    heave = table_rows(lines, "Table 1. The added-mass and damping coefficients of a heaving hemisphere")
    surge = table_rows(lines, "TABLE 2. The added-mass and damping coefficients of a surging hemisphere")
    full = [
        "# A. Hulme (1982), J. Fluid Mech. 121, 443-463, tables 1 and 2 (extracted by scripts/extract_hulme.py)",
        "# Ka A0 B0 A1 B1   (heave A(0), B(0); surge A(1), B(1); m0 = 2 pi rho a^3 / 3)",
    ]
    for ka in sorted((k for k in set(heave) & set(surge) if k != "inf"), key=float):
        if float(ka) > 0:
            full.append(f"{ka} {heave[ka][0]} {heave[ka][1]} {surge[ka][0]} {surge[ka][1]}")
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    OUT_FULL.write_text("\n".join(full) + "\n", encoding="ascii", newline="\n")
    print("\n".join(full))
    big_l = constant(text, r"L = 3 \\sum.*?= (\d\.\d+)\.\.\.")
    c1 = constant(text, r"C_1 = (\d\.\d+)\.\.\.")
    if abs(big_l - heave["0"][0]) > 5e-5 or abs(c1 - surge["inf"][0]) > 5e-5:
        raise SystemExit("the constants L, C1 do not agree with the rows 0 and inf of the tables")
    limits = [
        "# A. Hulme (1982), J. Fluid Mech. 121, 443-463: the added masses A(0) (heave) and A(1) (surge) / m0",
        "# at Ka = 0 and Ka = inf (extracted by scripts/extract_hulme.py): heave L of eq. (3.18) and the row inf",
        "# of table 1; surge the row 0 of table 2 (eq. 4.21) and C1 of eq. (5.15)",
        "# Ka A0 A1",
        f"0 {big_l} {surge['0'][0]}",
        f"inf {heave['inf'][0]} {c1}",
    ]
    OUT_LIMITS.write_text("\n".join(limits) + "\n", encoding="ascii", newline="\n")
    print("\n".join(limits))


if __name__ == "__main__":
    main()
